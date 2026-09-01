# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only readiness gate for normalized dashboard reads.

The checker takes one repeatable-read, read-only snapshot, fixes an as_of
cutoff, and reports whether eligible legacy data spans at least seven days and
the latest seven-day window has exact raw public-row and materialized-rollup
parity. Every parity query is time-bounded and the checker never writes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, NoReturn, Protocol

import psycopg

MINIMUM_HOURS = 168
DETAIL_LIMIT = 50
DEFAULT_CHUNK_HOURS = 1
DEFAULT_STATEMENT_TIMEOUT_SECONDS = 30
DEFAULT_TOTAL_TIMEOUT_SECONDS = 600
_BENCHMARKS = ("STT", "TTS", "S2S")


class Cursor(Protocol):
    def fetchone(self) -> tuple[Any, ...] | None: ...

    def fetchall(self) -> list[tuple[Any, ...]]: ...


class Connection(Protocol):
    def execute(self, query: str, params: Mapping[str, Any] | None = None) -> Cursor: ...


class ReadinessOperationalError(RuntimeError):
    """A bounded readiness check could not complete."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


_LATEST_ELIGIBLE_BUCKET_SQL = """
SELECT date_trunc('hour', COALESCE(run.scheduled_at, result.created_at)) AS bucket_at
FROM benchmarks_v2.results result
JOIN benchmarks_v2.runs run ON run.id = result.run_id
WHERE result.status = 'success'
  AND result.metric_value IS NOT NULL
  AND run.status IN ('succeeded', 'partial')
  AND result.benchmark = %(benchmark)s
  AND result.created_at < %(as_of)s
ORDER BY result.created_at DESC
LIMIT 1
"""

_HISTORICAL_COVERAGE_BUCKET_SQL = """
SELECT date_trunc('hour', COALESCE(run.scheduled_at, result.created_at)) AS bucket_at
FROM benchmarks_v2.results result
JOIN benchmarks_v2.runs run ON run.id = result.run_id
WHERE result.status = 'success'
  AND result.metric_value IS NOT NULL
  AND run.status IN ('succeeded', 'partial')
  AND result.benchmark = %(benchmark)s
  AND result.created_at < %(as_of)s
  AND result.created_at <= %(cutoff)s
  AND date_trunc('hour', COALESCE(run.scheduled_at, result.created_at)) <= %(cutoff)s
ORDER BY result.created_at DESC
LIMIT 1
"""

_NORMALIZED_DATASETS_SQL = """
SELECT DISTINCT observation.dataset_id
FROM benchmarks_v2.benchmark_observations observation
WHERE observation.status = 'succeeded'
  AND observation.benchmark = %(benchmark)s
  AND observation.captured_at >= %(start)s
  AND observation.captured_at < %(end)s
ORDER BY observation.dataset_id
"""

_ROLLUP_DATASETS_SQL = """
SELECT dataset_id
FROM benchmarks_v2.results_by_bucket
WHERE benchmark = %(benchmark)s
  AND bucket_at >= %(start)s
  AND bucket_at < %(end)s
GROUP BY dataset_id
UNION
SELECT dataset_id
FROM benchmarks_v2.metric_values_by_bucket
WHERE benchmark = %(benchmark)s
  AND metric_version = 'v1'
  AND evaluation_variant = 'default'
  AND value_key = 'primary'
  AND bucket_at >= %(start)s
  AND bucket_at < %(end)s
GROUP BY dataset_id
ORDER BY dataset_id
"""

# Both projections intentionally use the dashboard's public rows rather than
# normalized internals: WER components are attached to the public WER row and
# TTFA components regain their legacy public metric names.
_LEGACY_RAW_COUNTS_SQL = """
SELECT result.provider,
       result.model,
       result.benchmark,
       CASE WHEN result.benchmark = 'TTS' THEN 'tts-v1' ELSE run.dataset_id END AS dataset_id,
       result.metric_type,
       result.metric_value AS value,
       result.wer_insertions_pct,
       result.wer_deletions_pct,
       result.wer_substitutions_pct,
       COUNT(*) AS multiplicity
FROM benchmarks_v2.results result
JOIN benchmarks_v2.runs run ON run.id = result.run_id
WHERE result.status = 'success'
  AND result.metric_value IS NOT NULL
  AND run.status IN ('succeeded', 'partial')
  AND result.benchmark = %(benchmark)s
  AND result.created_at >= %(start)s
  AND result.created_at < %(end)s
GROUP BY result.provider,
         result.model,
         result.benchmark,
         CASE WHEN result.benchmark = 'TTS' THEN 'tts-v1' ELSE run.dataset_id END,
         result.metric_type,
         result.metric_value,
         result.wer_insertions_pct,
         result.wer_deletions_pct,
         result.wer_substitutions_pct
"""

_NORMALIZED_RAW_COUNTS_SQL = """
SELECT observation.provider,
       observation.model,
       observation.benchmark,
       observation.dataset_id,
       CASE
         WHEN evaluation.metric_type = 'TTFA' AND value.value_key = 'roundtrip'
           THEN 'TTFARoundtrip'
         WHEN evaluation.metric_type = 'TTFA' AND value.value_key = 'leading_silence'
           THEN 'TTFALeadingSilence'
         WHEN value.value_role = 'primary'
           THEN evaluation.metric_type
       END AS metric_type,
       value.value,
       components.wer_insertions_pct,
       components.wer_deletions_pct,
       components.wer_substitutions_pct,
       COUNT(*) AS multiplicity
FROM benchmarks_v2.benchmark_observations observation
JOIN benchmarks_v2.runs run ON run.id = observation.run_id
JOIN benchmarks_v2.metric_evaluations evaluation
  ON evaluation.observation_id = observation.id
JOIN benchmarks_v2.metric_values value
  ON value.metric_evaluation_id = evaluation.id
LEFT JOIN LATERAL (
  SELECT MAX(component.value) FILTER (
           WHERE component.value_key = 'insertions'
         ) AS wer_insertions_pct,
         MAX(component.value) FILTER (
           WHERE component.value_key = 'deletions'
         ) AS wer_deletions_pct,
         MAX(component.value) FILTER (
           WHERE component.value_key = 'substitutions'
         ) AS wer_substitutions_pct
  FROM benchmarks_v2.metric_values component
  WHERE component.metric_evaluation_id = evaluation.id
) components ON TRUE
WHERE observation.status = 'succeeded'
  AND evaluation.status = 'succeeded'
  AND run.status IN ('succeeded', 'partial')
  AND evaluation.metric_version = 'v1'
  AND evaluation.evaluation_variant = 'default'
  AND observation.benchmark = %(benchmark)s
  AND observation.dataset_id = %(dataset_id)s
  AND observation.captured_at >= %(start)s
  AND observation.captured_at < %(end)s
  AND (
    value.value_role = 'primary'
    OR (
      evaluation.metric_type = 'TTFA'
      AND value.value_key IN ('roundtrip', 'leading_silence')
    )
  )
GROUP BY observation.provider,
         observation.model,
         observation.benchmark,
         observation.dataset_id,
         CASE
           WHEN evaluation.metric_type = 'TTFA' AND value.value_key = 'roundtrip'
             THEN 'TTFARoundtrip'
           WHEN evaluation.metric_type = 'TTFA' AND value.value_key = 'leading_silence'
             THEN 'TTFALeadingSilence'
           WHEN value.value_role = 'primary'
             THEN evaluation.metric_type
         END,
         value.value,
         components.wer_insertions_pct,
         components.wer_deletions_pct,
         components.wer_substitutions_pct
"""

_LEGACY_ROLLUP_COUNTS_SQL = """
SELECT provider,
       model,
       benchmark,
       dataset_id,
       metric_type,
       bucket_at,
       min_value,
       p25,
       p50,
       p75,
       max_value,
       value_sum,
       sample_count,
       COUNT(*) AS multiplicity
FROM benchmarks_v2.results_by_bucket
WHERE benchmark = %(benchmark)s
  AND dataset_id = %(dataset_id)s
  AND bucket_at >= %(start)s
  AND bucket_at < %(end)s
GROUP BY provider,
         model,
         benchmark,
         dataset_id,
         metric_type,
         bucket_at,
         min_value,
         p25,
         p50,
         p75,
         max_value,
         value_sum,
         sample_count
"""

_NORMALIZED_ROLLUP_COUNTS_SQL = """
SELECT provider,
       model,
       benchmark,
       dataset_id,
       metric_type,
       bucket_at,
       min_value,
       p25,
       p50,
       p75,
       max_value,
       value_sum,
       sample_count,
       COUNT(*) AS multiplicity
FROM benchmarks_v2.metric_values_by_bucket
WHERE benchmark = %(benchmark)s
  AND dataset_id = %(dataset_id)s
  AND metric_version = 'v1'
  AND evaluation_variant = 'default'
  AND value_key = 'primary'
  AND bucket_at >= %(start)s
  AND bucket_at < %(end)s
GROUP BY provider,
         model,
         benchmark,
         dataset_id,
         metric_type,
         bucket_at,
         min_value,
         p25,
         p50,
         p75,
         max_value,
         value_sum,
         sample_count
"""


class _QueryRunner:
    def __init__(
        self,
        conn: Connection,
        *,
        statement_timeout_seconds: int,
        total_timeout_seconds: int,
    ) -> None:
        self._conn = conn
        self._statement_timeout_ms = statement_timeout_seconds * 1000
        self._deadline = time.monotonic() + total_timeout_seconds

    def execute(self, stage: str, query: str, params: Mapping[str, Any] | None = None) -> Cursor:
        remaining_ms = int((self._deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            raise ReadinessOperationalError(stage, "total readiness-check timeout exceeded")
        timeout_ms = max(1, min(self._statement_timeout_ms, remaining_ms))
        try:
            self._conn.execute(
                "SELECT set_config('statement_timeout', %(timeout)s, true)",
                {"timeout": f"{timeout_ms}ms"},
            )
            return self._conn.execute(query, params)
        except psycopg.Error as error:
            raise ReadinessOperationalError(stage, f"database statement failed: {error}") from error


def _coverage(buckets: Iterable[datetime]) -> tuple[float, datetime | None, datetime | None]:
    """Return the eligible data span without assuming a collection cadence."""
    present = sorted(set(buckets))
    if not present:
        return 0, None, None
    first, last = present[0], present[-1]
    return (last - first).total_seconds() / 3600, first, last


def _time_slices(
    start: datetime, end: datetime, chunk_hours: int
) -> Iterable[tuple[datetime, datetime]]:
    cursor = start
    delta = timedelta(hours=chunk_hours)
    while cursor < end:
        next_cursor = min(cursor + delta, end)
        yield cursor, next_cursor
        cursor = next_cursor


def _apply_grouped_counts(
    counter: Counter[tuple[Any, ...]],
    rows: Iterable[Sequence[Any]],
    *,
    sign: int,
) -> None:
    for row in rows:
        if not row:
            raise ReadinessOperationalError("parity", "count query returned an empty row")
        *public_row, multiplicity = row
        counter[tuple(public_row)] += sign * int(multiplicity)


def _row_sort_key(row: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((type(value).__name__, "" if value is None else str(value)) for value in row)


def _summarize_counter(
    counter: Counter[tuple[Any, ...]],
) -> tuple[int, list[tuple[Any, ...]]]:
    mismatch_count = sum(abs(count) for count in counter.values())
    details: list[tuple[Any, ...]] = []
    residuals = [
        ("legacy_only" if count > 0 else "normalized_only", row, abs(count))
        for row, count in counter.items()
        if count
    ]
    for side, row, multiplicity in sorted(
        residuals, key=lambda item: (item[0], _row_sort_key(item[1]))
    ):
        remaining = DETAIL_LIMIT - len(details)
        if remaining <= 0:
            break
        details.extend((side, *row) for _ in range(min(multiplicity, remaining)))
    return mismatch_count, details


def evaluate_readiness(
    *,
    as_of: datetime,
    buckets: Iterable[datetime],
    raw_mismatches: Sequence[Sequence[Any]],
    rollup_mismatches: Sequence[Sequence[Any]],
    required_hours: int = MINIMUM_HOURS,
    raw_mismatch_count: int | None = None,
    rollup_mismatch_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate snapshot rows without database access; used by focused tests."""
    coverage_hours, coverage_start, coverage_end = _coverage(buckets)
    raw_details = [list(row) for row in raw_mismatches[:DETAIL_LIMIT]]
    rollup_details = [list(row) for row in rollup_mismatches[:DETAIL_LIMIT]]
    raw_count = len(raw_mismatches) if raw_mismatch_count is None else raw_mismatch_count
    rollup_count = (
        len(rollup_mismatches) if rollup_mismatch_count is None else rollup_mismatch_count
    )
    report = {
        "as_of": as_of.isoformat(),
        "required_hours": required_hours,
        "coverage_hours": coverage_hours,
        "coverage_start": coverage_start.isoformat() if coverage_start else None,
        "coverage_end": coverage_end.isoformat() if coverage_end else None,
        "raw_mismatch_count": raw_count,
        "raw_mismatches": raw_details,
        "raw_mismatches_truncated": raw_count > len(raw_details),
        "rollup_mismatch_count": rollup_count,
        "rollup_mismatches": rollup_details,
        "rollup_mismatches_truncated": rollup_count > len(rollup_details),
    }
    report["ready"] = coverage_hours >= required_hours and not raw_count and not rollup_count
    return report


def _datasets_by_benchmark(
    runner: _QueryRunner,
    *,
    query: str,
    stage_prefix: str,
    start: datetime,
    end: datetime,
) -> dict[str, list[str]]:
    datasets: dict[str, list[str]] = {}
    for benchmark in _BENCHMARKS:
        rows = runner.execute(
            f"{stage_prefix}_datasets_{benchmark.lower()}",
            query,
            {"benchmark": benchmark, "start": start, "end": end},
        ).fetchall()
        datasets[benchmark] = [str(row[0]) for row in rows]
    return datasets


def check(
    conn: Connection,
    *,
    required_hours: int = MINIMUM_HOURS,
    chunk_hours: int = DEFAULT_CHUNK_HOURS,
    statement_timeout_seconds: int = DEFAULT_STATEMENT_TIMEOUT_SECONDS,
    total_timeout_seconds: int = DEFAULT_TOTAL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run every bounded read in one repeatable-read, read-only transaction."""
    if required_hours <= 0:
        raise ValueError("required_hours must be positive")
    if chunk_hours <= 0:
        raise ValueError("chunk_hours must be positive")
    if statement_timeout_seconds <= 0:
        raise ValueError("statement_timeout_seconds must be positive")
    if total_timeout_seconds <= 0:
        raise ValueError("total_timeout_seconds must be positive")

    try:
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
    except psycopg.Error as error:
        raise ReadinessOperationalError(
            "transaction", f"could not begin snapshot: {error}"
        ) from error

    runner = _QueryRunner(
        conn,
        statement_timeout_seconds=statement_timeout_seconds,
        total_timeout_seconds=total_timeout_seconds,
    )
    as_of_row = runner.execute("snapshot", "SELECT now()").fetchone()
    if as_of_row is None:
        raise ReadinessOperationalError("snapshot", "SELECT now() returned no row")
    as_of = as_of_row[0]
    start = as_of - timedelta(hours=required_hours)

    latest_candidates: list[datetime] = []
    for benchmark in _BENCHMARKS:
        latest_row = runner.execute(
            f"coverage_latest_{benchmark.lower()}",
            _LATEST_ELIGIBLE_BUCKET_SQL,
            {"benchmark": benchmark, "as_of": as_of},
        ).fetchone()
        if latest_row is not None:
            latest_candidates.append(latest_row[0])

    buckets: list[datetime] = []
    if latest_candidates:
        latest_bucket = max(latest_candidates)
        cutoff = latest_bucket - timedelta(hours=required_hours)
        historical_candidates: list[datetime] = []
        for benchmark in _BENCHMARKS:
            historical_row = runner.execute(
                f"coverage_historical_{benchmark.lower()}",
                _HISTORICAL_COVERAGE_BUCKET_SQL,
                {"benchmark": benchmark, "as_of": as_of, "cutoff": cutoff},
            ).fetchone()
            if historical_row is not None:
                historical_candidates.append(historical_row[0])
        buckets = [latest_bucket]
        if historical_candidates:
            buckets.insert(0, min(historical_candidates))

    normalized_datasets = _datasets_by_benchmark(
        runner,
        query=_NORMALIZED_DATASETS_SQL,
        stage_prefix="raw",
        start=start,
        end=as_of,
    )
    rollup_datasets = _datasets_by_benchmark(
        runner,
        query=_ROLLUP_DATASETS_SQL,
        stage_prefix="rollup",
        start=start,
        end=as_of,
    )

    raw_counter: Counter[tuple[Any, ...]] = Counter()
    rollup_counter: Counter[tuple[Any, ...]] = Counter()
    for slice_start, slice_end in _time_slices(start, as_of, chunk_hours):
        window = {"start": slice_start, "end": slice_end}
        slice_label = slice_start.isoformat()
        for benchmark in _BENCHMARKS:
            _apply_grouped_counts(
                raw_counter,
                runner.execute(
                    f"raw_legacy_{benchmark.lower()}_{slice_label}",
                    _LEGACY_RAW_COUNTS_SQL,
                    {**window, "benchmark": benchmark},
                ).fetchall(),
                sign=1,
            )
            for dataset_id in normalized_datasets[benchmark]:
                _apply_grouped_counts(
                    raw_counter,
                    runner.execute(
                        f"raw_normalized_{benchmark.lower()}_{slice_label}",
                        _NORMALIZED_RAW_COUNTS_SQL,
                        {**window, "benchmark": benchmark, "dataset_id": dataset_id},
                    ).fetchall(),
                    sign=-1,
                )

            for dataset_id in rollup_datasets[benchmark]:
                cohort = {**window, "benchmark": benchmark, "dataset_id": dataset_id}
                _apply_grouped_counts(
                    rollup_counter,
                    runner.execute(
                        f"rollup_legacy_{benchmark.lower()}_{slice_label}",
                        _LEGACY_ROLLUP_COUNTS_SQL,
                        cohort,
                    ).fetchall(),
                    sign=1,
                )
                _apply_grouped_counts(
                    rollup_counter,
                    runner.execute(
                        f"rollup_normalized_{benchmark.lower()}_{slice_label}",
                        _NORMALIZED_ROLLUP_COUNTS_SQL,
                        cohort,
                    ).fetchall(),
                    sign=-1,
                )

    raw_count, raw_details = _summarize_counter(raw_counter)
    rollup_count, rollup_details = _summarize_counter(rollup_counter)
    return evaluate_readiness(
        as_of=as_of,
        buckets=buckets,
        raw_mismatches=raw_details,
        rollup_mismatches=rollup_details,
        required_hours=required_hours,
        raw_mismatch_count=raw_count,
        rollup_mismatch_count=rollup_count,
    )


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ReadinessOperationalError("configuration", message)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _JsonArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--chunk-hours", type=_positive_int, default=DEFAULT_CHUNK_HOURS)
    parser.add_argument(
        "--statement-timeout-seconds",
        type=_positive_int,
        default=DEFAULT_STATEMENT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--total-timeout-seconds",
        type=_positive_int,
        default=DEFAULT_TOTAL_TIMEOUT_SECONDS,
    )
    return parser.parse_args(argv)


def _error_report(stage: str, error: object) -> dict[str, Any]:
    return {"ready": False, "stage": stage, "error": str(error)}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _args(argv)
    except ReadinessOperationalError as error:
        print(json.dumps(_error_report(error.stage, error), sort_keys=True))
        return 2
    if not args.database_url:
        print(
            json.dumps(_error_report("configuration", "DATABASE_URL or --database-url is required"))
        )
        return 2
    try:
        with psycopg.connect(args.database_url, autocommit=False) as conn:
            report = check(
                conn,
                chunk_hours=args.chunk_hours,
                statement_timeout_seconds=args.statement_timeout_seconds,
                total_timeout_seconds=args.total_timeout_seconds,
            )
    except ReadinessOperationalError as error:
        print(json.dumps(_error_report(error.stage, error), sort_keys=True))
        return 2
    except psycopg.Error as error:
        print(json.dumps(_error_report("connection", error), sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True, default=str))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
