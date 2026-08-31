# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only readiness gate for normalized dashboard reads.

The checker takes one repeatable-read, read-only snapshot, fixes an ``as_of``
cutoff, and reports whether eligible legacy data spans at least seven days and
the latest seven-day window has exact raw public-row and materialized-rollup
parity. It never writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol

import psycopg

MINIMUM_HOURS = 168
DETAIL_LIMIT = 50


class Cursor(Protocol):
    def fetchone(self) -> tuple[Any, ...] | None: ...

    def fetchall(self) -> list[tuple[Any, ...]]: ...


class Connection(Protocol):
    def execute(self, query: str, params: Mapping[str, Any] | None = None) -> Cursor: ...


_ELIGIBLE_BUCKETS_SQL = """
SELECT DISTINCT date_trunc('hour', COALESCE(run.scheduled_at, result.created_at)) AS bucket_at
FROM benchmarks_v2.results result JOIN benchmarks_v2.runs run ON run.id = result.run_id
WHERE result.status = 'success' AND result.metric_value IS NOT NULL
  AND run.status IN ('succeeded', 'partial') AND result.created_at < %(as_of)s
ORDER BY bucket_at
"""

# Both projections intentionally use the dashboard's public rows rather than
# normalized internals: WER components are omitted and TTFA components regain
# their legacy public metric names.
_RAW_MISMATCH_SQL = """
WITH legacy AS (
 SELECT result.provider, result.model, result.benchmark,
        CASE WHEN result.benchmark = 'TTS' THEN 'tts-v1' ELSE run.dataset_id END AS dataset_id,
        result.metric_type, result.metric_value AS value, result.wer_insertions_pct,
        result.wer_deletions_pct, result.wer_substitutions_pct
 FROM benchmarks_v2.results result JOIN benchmarks_v2.runs run ON run.id = result.run_id
 WHERE result.status = 'success' AND result.metric_value IS NOT NULL
   AND run.status IN ('succeeded', 'partial') AND result.created_at >= %(start)s
   AND result.created_at < %(as_of)s
), normalized AS (
 SELECT observation.provider, observation.model, observation.benchmark, observation.dataset_id,
        CASE WHEN evaluation.metric_type = 'TTFA' AND value.value_key = 'roundtrip'
                  THEN 'TTFARoundtrip'
             WHEN evaluation.metric_type = 'TTFA' AND value.value_key = 'leading_silence'
                  THEN 'TTFALeadingSilence'
             WHEN value.value_role = 'primary' THEN evaluation.metric_type END AS metric_type,
        value.value, insertion.value AS wer_insertions_pct,
        deletion.value AS wer_deletions_pct, substitution.value AS wer_substitutions_pct
 FROM benchmarks_v2.metric_values value
 JOIN benchmarks_v2.metric_evaluations evaluation ON evaluation.id = value.metric_evaluation_id
 JOIN benchmarks_v2.benchmark_observations observation ON observation.id = evaluation.observation_id
 JOIN benchmarks_v2.runs run ON run.id = observation.run_id
 LEFT JOIN benchmarks_v2.metric_values insertion
   ON insertion.metric_evaluation_id = evaluation.id AND insertion.value_key = 'insertions'
 LEFT JOIN benchmarks_v2.metric_values deletion
   ON deletion.metric_evaluation_id = evaluation.id AND deletion.value_key = 'deletions'
 LEFT JOIN benchmarks_v2.metric_values substitution
   ON substitution.metric_evaluation_id = evaluation.id AND substitution.value_key = 'substitutions'
 WHERE observation.status = 'succeeded' AND evaluation.status = 'succeeded'
   AND run.status IN ('succeeded', 'partial') AND evaluation.metric_version = 'v1'
   AND evaluation.evaluation_variant = 'default' AND observation.captured_at >= %(start)s
   AND observation.captured_at < %(as_of)s AND (value.value_role = 'primary'
        OR (evaluation.metric_type = 'TTFA'
            AND value.value_key IN ('roundtrip', 'leading_silence')))
), mismatches AS (
 (SELECT 'legacy_only' AS side, * FROM legacy
  EXCEPT ALL
  SELECT 'legacy_only', provider, model, benchmark, dataset_id, metric_type, value,
         wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct
  FROM normalized WHERE metric_type IS NOT NULL)
 UNION ALL
 (SELECT 'normalized_only', provider, model, benchmark, dataset_id, metric_type, value,
         wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct
  FROM normalized WHERE metric_type IS NOT NULL
  EXCEPT ALL
  SELECT 'normalized_only', * FROM legacy)
) SELECT * FROM mismatches
ORDER BY side, benchmark, dataset_id, provider, model, metric_type, value
"""

_ROLLUP_MISMATCH_SQL = """
WITH legacy AS (
 SELECT provider, model, benchmark, dataset_id, metric_type, bucket_at,
        min_value, p25, p50, p75, max_value, value_sum, sample_count
 FROM benchmarks_v2.results_by_bucket WHERE bucket_at >= %(start)s AND bucket_at < %(as_of)s
), normalized AS (
 SELECT provider, model, benchmark, dataset_id, metric_type, bucket_at,
        min_value, p25, p50, p75, max_value, value_sum, sample_count
 FROM benchmarks_v2.metric_values_by_bucket
 WHERE metric_version = 'v1' AND evaluation_variant = 'default'
   AND value_key = 'primary'
   AND bucket_at >= %(start)s AND bucket_at < %(as_of)s
), mismatches AS (
 (SELECT 'legacy_only' AS side, * FROM legacy
  EXCEPT ALL SELECT 'legacy_only', * FROM normalized)
 UNION ALL
 (SELECT 'normalized_only' AS side, * FROM normalized
  EXCEPT ALL SELECT 'normalized_only', * FROM legacy)
) SELECT * FROM mismatches
ORDER BY side, benchmark, dataset_id, provider, model, metric_type, bucket_at
"""


def _coverage(buckets: Iterable[datetime]) -> tuple[float, datetime | None, datetime | None]:
    """Return the eligible data span without assuming a collection cadence."""
    present = sorted(set(buckets))
    if not present:
        return 0, None, None
    first, last = present[0], present[-1]
    return (last - first).total_seconds() / 3600, first, last


def evaluate_readiness(
    *,
    as_of: datetime,
    buckets: Iterable[datetime],
    raw_mismatches: Sequence[Sequence[Any]],
    rollup_mismatches: Sequence[Sequence[Any]],
    required_hours: int = MINIMUM_HOURS,
) -> dict[str, Any]:
    """Evaluate snapshot rows without database access; used by focused tests."""
    coverage_hours, coverage_start, coverage_end = _coverage(buckets)
    raw_count, rollup_count = len(raw_mismatches), len(rollup_mismatches)
    report = {
        "as_of": as_of.isoformat(),
        "required_hours": required_hours,
        "coverage_hours": coverage_hours,
        "coverage_start": coverage_start.isoformat() if coverage_start else None,
        "coverage_end": coverage_end.isoformat() if coverage_end else None,
        "raw_mismatch_count": raw_count,
        "raw_mismatches": [list(row) for row in raw_mismatches[:DETAIL_LIMIT]],
        "raw_mismatches_truncated": raw_count > DETAIL_LIMIT,
        "rollup_mismatch_count": rollup_count,
        "rollup_mismatches": [list(row) for row in rollup_mismatches[:DETAIL_LIMIT]],
        "rollup_mismatches_truncated": rollup_count > DETAIL_LIMIT,
    }
    report["ready"] = coverage_hours >= required_hours and not raw_count and not rollup_count
    return report


def check(conn: Connection, *, required_hours: int = MINIMUM_HOURS) -> dict[str, Any]:
    """Run every read in exactly one repeatable-read read-only transaction."""
    conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
    as_of_row = conn.execute("SELECT now()").fetchone()
    if as_of_row is None:
        raise RuntimeError("SELECT now() returned no row")
    as_of = as_of_row[0]
    start = as_of - timedelta(hours=required_hours)
    params = {"start": start, "as_of": as_of}
    buckets = [row[0] for row in conn.execute(_ELIGIBLE_BUCKETS_SQL, params).fetchall()]
    raw = conn.execute(_RAW_MISMATCH_SQL, params).fetchall()
    rollups = conn.execute(_ROLLUP_MISMATCH_SQL, params).fetchall()
    return evaluate_readiness(
        as_of=as_of,
        buckets=buckets,
        raw_mismatches=raw,
        rollup_mismatches=rollups,
        required_hours=required_hours,
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.database_url:
        print("DATABASE_URL or --database-url is required", file=sys.stderr)
        return 2
    with psycopg.connect(args.database_url, autocommit=False) as conn:
        report = check(conn)
    print(json.dumps(report, sort_keys=True, default=str))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
