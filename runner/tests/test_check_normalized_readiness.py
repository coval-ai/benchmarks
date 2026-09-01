# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bounded normalized dashboard readiness gate."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, NoReturn

import psycopg
import pytest

import scripts.check_normalized_readiness as readiness
from scripts.check_normalized_readiness import (
    DETAIL_LIMIT,
    MINIMUM_HOURS,
    ReadinessOperationalError,
    check,
    evaluate_readiness,
)

AS_OF = datetime(2026, 8, 31, 12, tzinfo=UTC)
RAW_ROW = (
    "provider",
    "model",
    "STT",
    "stt-v1",
    "WER",
    3.0,
    1.0,
    1.0,
    1.0,
)
ROLLUP_ROW = (
    "provider",
    "model",
    "STT",
    "stt-v1",
    "WER",
    AS_OF - timedelta(hours=2),
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1,
)


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
        raw_mismatches=[("legacy_only", *RAW_ROW)],
        rollup_mismatches=[],
    )
    assert report["ready"] is False
    assert report["raw_mismatch_count"] == 1


def test_readiness_rejects_rollup_mismatch() -> None:
    report = evaluate_readiness(
        as_of=AS_OF,
        buckets=_buckets(),
        raw_mismatches=[],
        rollup_mismatches=[("normalized_only", *ROLLUP_ROW)],
    )
    assert report["ready"] is False
    assert report["rollup_mismatch_count"] == 1


def test_grouped_counts_cancel_globally_across_slices() -> None:
    counter: Counter[tuple[Any, ...]] = Counter()
    readiness._apply_grouped_counts(counter, [(*RAW_ROW, 1)], sign=1)
    readiness._apply_grouped_counts(counter, [(*RAW_ROW, 1)], sign=-1)
    count, details = readiness._summarize_counter(counter)
    assert count == 0
    assert details == []
    assert not counter


def test_duplicate_residual_count_is_exact_while_details_are_bounded() -> None:
    counter: Counter[tuple[Any, ...]] = Counter()
    readiness._apply_grouped_counts(counter, [(*RAW_ROW, DETAIL_LIMIT + 7)], sign=1)
    readiness._apply_grouped_counts(counter, [(*RAW_ROW, 2)], sign=-1)

    count, details = readiness._summarize_counter(counter)
    report = evaluate_readiness(
        as_of=AS_OF,
        buckets=_buckets(),
        raw_mismatches=details,
        rollup_mismatches=[],
        raw_mismatch_count=count,
    )

    assert count == DETAIL_LIMIT + 5
    assert len(details) == DETAIL_LIMIT
    assert {row[0] for row in details} == {"legacy_only"}
    assert report["raw_mismatch_count"] == DETAIL_LIMIT + 5
    assert report["raw_mismatches_truncated"] is True


def test_normalized_only_residual_has_the_correct_side() -> None:
    counter: Counter[tuple[Any, ...]] = Counter()
    readiness._apply_grouped_counts(counter, [(*RAW_ROW, 2)], sign=-1)
    count, details = readiness._summarize_counter(counter)
    assert count == 2
    assert details == [("normalized_only", *RAW_ROW), ("normalized_only", *RAW_ROW)]


def test_time_slices_are_half_open_and_cover_a_partial_final_chunk() -> None:
    start = AS_OF - timedelta(hours=2, minutes=30)
    slices = list(readiness._time_slices(start, AS_OF, 1))
    assert slices == [
        (start, start + timedelta(hours=1)),
        (start + timedelta(hours=1), start + timedelta(hours=2)),
        (start + timedelta(hours=2), AS_OF),
    ]
    assert all(left < right for left, right in slices)
    assert all(slices[index][1] == slices[index + 1][0] for index in range(len(slices) - 1))


def test_sql_preserves_public_mapping_and_uses_bounded_counts() -> None:
    normalized = readiness._NORMALIZED_RAW_COUNTS_SQL
    assert "TTFARoundtrip" in normalized
    assert "TTFALeadingSilence" in normalized
    assert "LEFT JOIN LATERAL" in normalized
    assert "value_key = 'insertions'" in normalized
    assert "observation.benchmark = %(benchmark)s" in normalized
    assert "observation.dataset_id = %(dataset_id)s" in normalized
    assert "observation.captured_at >= %(start)s" in normalized
    assert "observation.captured_at < %(end)s" in normalized
    assert "COUNT(*) AS multiplicity" in normalized
    assert "EXCEPT ALL" not in normalized

    legacy = readiness._LEGACY_RAW_COUNTS_SQL
    assert "result.benchmark = %(benchmark)s" in legacy
    assert "result.created_at >= %(start)s" in legacy
    assert "result.created_at < %(end)s" in legacy

    latest_coverage = readiness._LATEST_ELIGIBLE_BUCKET_SQL
    assert "result.benchmark = %(benchmark)s" in latest_coverage
    assert "ORDER BY result.created_at DESC" in latest_coverage
    assert "LIMIT 1" in latest_coverage
    assert "MIN(" not in latest_coverage
    assert "MAX(" not in latest_coverage

    historical_coverage = readiness._HISTORICAL_COVERAGE_BUCKET_SQL
    assert "result.created_at <= %(cutoff)s" in historical_coverage
    assert "COALESCE(run.scheduled_at, result.created_at)) <= %(cutoff)s" in historical_coverage
    assert "LIMIT 1" in historical_coverage


class _Cursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class _Connection:
    def __init__(
        self,
        *,
        legacy_raw_at: datetime | None = None,
        normalized_raw_at: datetime | None = None,
        legacy_rollup_at: datetime | None = None,
        normalized_rollup_at: datetime | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.legacy_raw_at = legacy_raw_at
        self.normalized_raw_at = normalized_raw_at
        self.legacy_rollup_at = legacy_rollup_at
        self.normalized_rollup_at = normalized_rollup_at

    def execute(self, query: str, params: Mapping[str, Any] | None = None) -> _Cursor:
        copied_params = dict(params or {})
        self.calls.append((query, copied_params))
        if query.startswith("BEGIN") or "set_config('statement_timeout'" in query:
            return _Cursor([])
        if query == "SELECT now()":
            return _Cursor([(AS_OF,)])
        if query == readiness._LATEST_ELIGIBLE_BUCKET_SQL:
            return _Cursor([(AS_OF,)]) if copied_params["benchmark"] == "STT" else _Cursor([])
        if query == readiness._HISTORICAL_COVERAGE_BUCKET_SQL:
            return (
                _Cursor([(AS_OF - timedelta(days=7),)])
                if copied_params["benchmark"] == "STT"
                else _Cursor([])
            )
        if query == readiness._NORMALIZED_DATASETS_SQL:
            normalized_dataset_rows: list[tuple[Any, ...]] = (
                [("stt-v1",)]
                if copied_params["benchmark"] == "STT" and self.normalized_raw_at is not None
                else []
            )
            return _Cursor(normalized_dataset_rows)
        if query == readiness._ROLLUP_DATASETS_SQL:
            rollup_dataset_rows: list[tuple[Any, ...]] = (
                [("stt-v1",)]
                if copied_params["benchmark"] == "STT"
                and (self.legacy_rollup_at is not None or self.normalized_rollup_at is not None)
                else []
            )
            return _Cursor(rollup_dataset_rows)
        if query == readiness._LEGACY_RAW_COUNTS_SQL:
            legacy_raw_rows: list[tuple[Any, ...]] = (
                [(*RAW_ROW, 1)]
                if copied_params["benchmark"] == "STT"
                and copied_params["start"] == self.legacy_raw_at
                else []
            )
            return _Cursor(legacy_raw_rows)
        if query == readiness._NORMALIZED_RAW_COUNTS_SQL:
            normalized_raw_rows = (
                [(*RAW_ROW, 1)] if copied_params["start"] == self.normalized_raw_at else []
            )
            return _Cursor(normalized_raw_rows)
        if query == readiness._LEGACY_ROLLUP_COUNTS_SQL:
            legacy_rollup_rows = (
                [(*ROLLUP_ROW, 1)] if copied_params["start"] == self.legacy_rollup_at else []
            )
            return _Cursor(legacy_rollup_rows)
        if query == readiness._NORMALIZED_ROLLUP_COUNTS_SQL:
            normalized_rollup_rows = (
                [(*ROLLUP_ROW, 1)] if copied_params["start"] == self.normalized_rollup_at else []
            )
            return _Cursor(normalized_rollup_rows)
        raise AssertionError(f"unexpected query: {query}")


def test_checker_uses_one_snapshot_and_cancels_cross_slice_raw_rows() -> None:
    conn = _Connection(
        legacy_raw_at=AS_OF - timedelta(hours=2),
        normalized_raw_at=AS_OF - timedelta(hours=1),
        legacy_rollup_at=AS_OF - timedelta(hours=2),
        normalized_rollup_at=AS_OF - timedelta(hours=2),
    )
    report = check(conn, required_hours=2, chunk_hours=1)

    assert report["ready"] is True
    begin_calls = [query for query, _ in conn.calls if query.startswith("BEGIN")]
    assert begin_calls == ["BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"]

    target_calls = [
        query
        for query, _ in conn.calls
        if not query.startswith("BEGIN") and "set_config('statement_timeout'" not in query
    ]
    timeout_calls = [query for query, _ in conn.calls if "set_config('statement_timeout'" in query]
    assert len(timeout_calls) == len(target_calls)

    legacy_windows = [
        params for query, params in conn.calls if query == readiness._LEGACY_RAW_COUNTS_SQL
    ]
    assert len(legacy_windows) == 6
    assert all(params["end"] - params["start"] == timedelta(hours=1) for params in legacy_windows)
    assert any(
        query == readiness._NORMALIZED_RAW_COUNTS_SQL and params["dataset_id"] == "stt-v1"
        for query, params in conn.calls
    )


def test_checker_detects_normalized_only_cohort() -> None:
    conn = _Connection(normalized_raw_at=AS_OF - timedelta(hours=1))
    report = check(conn, required_hours=2, chunk_hours=1)
    assert report["ready"] is False
    assert report["raw_mismatch_count"] == 1
    assert report["raw_mismatches"][0][0] == "normalized_only"


def test_checker_detects_rollup_only_mismatch() -> None:
    conn = _Connection(legacy_rollup_at=AS_OF - timedelta(hours=2))
    report = check(conn, required_hours=2, chunk_hours=1)
    assert report["ready"] is False
    assert report["rollup_mismatch_count"] == 1
    assert report["rollup_mismatches"][0][0] == "legacy_only"


def test_total_deadline_fails_before_an_unbounded_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    conn = _Connection()
    runner = readiness._QueryRunner(conn, statement_timeout_seconds=30, total_timeout_seconds=1)

    with pytest.raises(ReadinessOperationalError, match="total readiness-check timeout") as raised:
        runner.execute("raw_legacy_stt", readiness._LEGACY_RAW_COUNTS_SQL, {})
    assert raised.value.stage == "raw_legacy_stt"
    assert conn.calls == []


@pytest.mark.parametrize(
    "keyword",
    ["required_hours", "chunk_hours", "statement_timeout_seconds", "total_timeout_seconds"],
)
def test_checker_rejects_non_positive_limits(keyword: str) -> None:
    kwargs = {
        "required_hours": 1,
        "chunk_hours": 1,
        "statement_timeout_seconds": 1,
        "total_timeout_seconds": 1,
    }
    kwargs[keyword] = 0
    with pytest.raises(ValueError, match="positive"):
        check(_Connection(), **kwargs)


def test_main_reports_connection_failure_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_connect(*args: Any, **kwargs: Any) -> NoReturn:
        raise psycopg.OperationalError("database unavailable")

    monkeypatch.setattr(psycopg, "connect", fail_connect)
    assert readiness.main(["--database-url", "postgresql://invalid"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "error": "database unavailable",
        "ready": False,
        "stage": "connection",
    }


@pytest.mark.parametrize("value", ["0", "not-an-integer"])
def test_cli_reports_invalid_chunk_size_as_json(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert readiness.main(["--database-url", "postgresql://unused", "--chunk-hours", value]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False
    assert report["stage"] == "configuration"
    assert "chunk-hours" in report["error"]
