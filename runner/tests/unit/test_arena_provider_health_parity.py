# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the read-only arena legacy/normalized parity report."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from psycopg import Connection

from coval_bench.arena.provider_health_parity import (
    _summarize,
    audit_provider_health_parity,
)


class _Cursor:
    def __init__(self, legacy: list[dict[str, Any]], normalized: list[dict[str, Any]]) -> None:
        self._legacy = legacy
        self._normalized = normalized
        self._rows: list[dict[str, Any]] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: str, _params: dict[str, Any]) -> None:
        self._rows = self._legacy if "benchmarks_v2.results" in statement else self._normalized

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class _Transaction:
    def __enter__(self) -> _Transaction:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Connection:
    def __init__(self, legacy: list[dict[str, Any]], normalized: list[dict[str, Any]]) -> None:
        self._legacy = legacy
        self._normalized = normalized

    def transaction(self) -> _Transaction:
        return _Transaction()

    def execute(self, _statement: str) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _Cursor:
        del row_factory
        return _Cursor(self._legacy, self._normalized)


def _connection(legacy: list[dict[str, Any]], normalized: list[dict[str, Any]]) -> Connection[Any]:
    # The fake implements only the connection operations exercised by this audit.
    return cast(Connection[Any], _Connection(legacy, normalized))


def test_summarize_counts_all_terminal_classifications_without_error_bodies() -> None:
    report = _summarize(
        [
            {"provider": "p", "run_id": 7, "status": "succeeded", "error": None},
            {"provider": "p", "run_id": 7, "status": "failed", "error": "invalid api key"},
            {"provider": "p", "run_id": 7, "status": "failed", "error": "HTTP 429"},
            {"provider": "p", "run_id": 7, "status": "failed", "error": "upstream"},
        ]
    )
    assert report["p"]["succeeded"] == 1
    assert report["p"]["failed"] == 3
    assert report["p"]["classified"] == {
        "credit": 0,
        "auth": 1,
        "rate_limit": 1,
        "other": 1,
    }
    assert report["p"]["benched"] is False
    assert all("error" not in value for value in report["p"].values() if isinstance(value, dict))


def test_rate_limit_and_other_failures_do_not_bench() -> None:
    for error in ("HTTP 429", "upstream exploded"):
        report = _summarize([{"provider": "p", "run_id": 1, "status": "failed", "error": error}])
        assert report["p"]["benched"] is False


def test_audit_redacts_errors_and_reports_benched_exclusion_sets() -> None:
    legacy = [
        {
            "provider": "dead",
            "run_id": 11,
            "status": "failed",
            "error": "insufficient credit secret key",
        },
    ]
    normalized = [
        {"provider": "dead", "run_id": 11, "status": "failed", "error": "invalid api key"},
        {"provider": "healthy", "run_id": 12, "status": "succeeded", "error": None},
    ]
    report = audit_provider_health_parity(
        _connection(legacy, normalized),
        as_of=datetime(2026, 9, 21, tzinfo=UTC),
        sample_limit=1,
    )
    assert report["complete"] is True
    assert report["excluded_sets"] == {"legacy_only": [], "normalized_only": []}
    assert report["mismatch_count"] == 2
    assert report["mismatches"] == [{"provider": "dead", "kind": "population_or_decision"}]
    assert "secret key" not in repr(report)


def test_audit_marks_a_row_limit_as_incomplete() -> None:
    rows = [{"provider": "p", "run_id": 1, "status": "failed", "error": "nope"}]
    report = audit_provider_health_parity(_connection(rows, rows), row_limit=1)
    assert report["complete"] is True
    report = audit_provider_health_parity(_connection(rows + rows, rows), row_limit=1)
    assert report["complete"] is False
    assert report["mismatches_truncated"] is True
