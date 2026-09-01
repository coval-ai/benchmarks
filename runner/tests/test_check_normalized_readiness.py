# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the read-only normalized dashboard readiness gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from scripts.check_normalized_readiness import MINIMUM_HOURS, check, evaluate_readiness

AS_OF = datetime(2026, 8, 31, 12, tzinfo=UTC)


def _buckets(hours: int = MINIMUM_HOURS) -> list[datetime]:
    return [AS_OF - timedelta(hours=hours), AS_OF]


def test_readiness_passes_with_seven_day_span_and_parity() -> None:
    report = evaluate_readiness(
        as_of=AS_OF, buckets=_buckets(), raw_mismatches=[], rollup_mismatches=[]
    )
    assert report["ready"] is True
    assert report["coverage_hours"] == MINIMUM_HOURS


def test_readiness_rejects_insufficient_span() -> None:
    report = evaluate_readiness(
        as_of=AS_OF, buckets=_buckets(MINIMUM_HOURS - 1), raw_mismatches=[], rollup_mismatches=[]
    )
    assert report["ready"] is False
    assert report["coverage_hours"] == MINIMUM_HOURS - 1


def test_readiness_does_not_assume_an_hourly_schedule() -> None:
    buckets = [AS_OF - timedelta(days=7), AS_OF - timedelta(days=3), AS_OF]
    report = evaluate_readiness(
        as_of=AS_OF, buckets=buckets, raw_mismatches=[], rollup_mismatches=[]
    )
    assert report["ready"] is True


def test_readiness_rejects_public_row_mismatch() -> None:
    report = evaluate_readiness(
        as_of=AS_OF,
        buckets=_buckets(),
        raw_mismatches=[("legacy_only", "provider", "model", "STT", "stt-v1", "WER", 3.0)],
        rollup_mismatches=[],
    )
    assert report["ready"] is False
    assert report["raw_mismatch_count"] == 1


def test_readiness_rejects_rollup_mismatch() -> None:
    report = evaluate_readiness(
        as_of=AS_OF,
        buckets=_buckets(),
        raw_mismatches=[],
        rollup_mismatches=[("normalized_only", "provider", "model")],
    )
    assert report["ready"] is False
    assert report["rollup_mismatch_count"] == 1


class _Cursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows

    def fetchone(self) -> tuple[Any, ...]:
        return self.rows[0]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class _Connection:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.responses = [
            [(AS_OF,)],
            [(bucket,) for bucket in _buckets()],
            [],
            [],
        ]

    def execute(self, query: str, params: Any = None) -> _Cursor:
        self.calls.append(query)
        if query.startswith("BEGIN"):
            return _Cursor([])
        return _Cursor(self.responses.pop(0))


def test_checker_uses_one_repeatable_read_only_transaction() -> None:
    conn = _Connection()
    report = check(conn)
    assert report["ready"] is True
    assert conn.calls[0] == "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"
    assert len(conn.calls) == 5
