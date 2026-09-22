# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, ANN401, B905, SIM117, S608
"""Read-only, full-history parity audit for the normalized S2S dedup reader.

This module intentionally accepts an already-configured psycopg connection (or
DSN through :func:`audit_s2s_dsn`).  It does not load application settings or
``.env`` files.  All queries run in one repeatable-read, read-only transaction
and return aggregate counts plus a bounded sample of identity mismatches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus

_DEFAULT_SAMPLE_LIMIT = 100
_STATEMENT_TIMEOUT = "30s"


@dataclass(frozen=True)
class S2SAuditReport:
    """Machine-readable parity result suitable for an operator wrapper."""

    bounds: dict[str, Any]
    row_counts: dict[str, int]
    key_counts: dict[str, int]
    mismatches: dict[str, Any]
    failed_null_metric_coverage: dict[str, int]
    unusable_identities: dict[str, int]
    explain: Any | None = None
    sample_limit: int = _DEFAULT_SAMPLE_LIMIT

    @property
    def ready(self) -> bool:
        return (
            self.key_counts.get("legacy_eligible_keys", 0) > 0
            and self.mismatches.get("legacy_missing_from_normalized_keys", 0) == 0
            and self.mismatches.get("normalized_extra_keys", 0) == 0
            and not any(self.unusable_identities.values())
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ready"] = self.ready
        result["complete"] = True
        return result


@dataclass
class _AuditState:
    bounds: dict[str, Any] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    key_counts: dict[str, int] = field(default_factory=dict)
    mismatches: dict[str, Any] = field(default_factory=dict)
    failed_null_metric_coverage: dict[str, int] = field(default_factory=dict)
    unusable_identities: dict[str, int] = field(default_factory=dict)
    explain: Any | None = None


def _required(value: Any | None) -> Any:
    if value is None:
        raise RuntimeError("audit query returned no evidence")
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def audit_s2s(
    conn: psycopg.Connection[Any],
    *,
    sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
    include_explain: bool = False,
) -> S2SAuditReport:
    """Audit every retained S2S row visible to ``conn``.

    The audit considers only succeeded/partial parent runs for dedup parity,
    matching the runtime reader, while separately reporting failed/null metric
    coverage across all parent statuses.  It never filters by time, model, or
    visibility and never returns raw result payloads.
    """
    if not 0 <= sample_limit <= 1000:
        raise ValueError("sample_limit must be between 0 and 1000")
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError("audit requires an idle connection")
    state = _AuditState()
    with conn.transaction():
        with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cur.execute("SELECT set_config('statement_timeout', %s, true)", (_STATEMENT_TIMEOUT,))
            cur.execute("SET LOCAL lock_timeout = '3s'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout = '30s'")

            cur.execute(
                """
                SELECT count(*)::bigint, min(r.id), max(r.id), min(r.created_at), max(r.created_at),
                       count(DISTINCT r.run_id)::bigint
                FROM benchmarks_v2.results r
                WHERE r.benchmark = 'S2S'
                """
            )
            row_count, min_id, max_id, min_created, max_created, run_count = _required(
                cur.fetchone()
            )
            state.bounds = {
                "legacy_result_id_min": min_id,
                "legacy_result_id_max": max_id,
                "legacy_created_at_min": _jsonable(min_created),
                "legacy_created_at_max": _jsonable(max_created),
                "legacy_run_count": int(run_count),
            }
            cur.execute(
                "SELECT transaction_timestamp(), current_setting('transaction_isolation'), current_setting('transaction_read_only')"
            )
            snapshot_at, isolation, read_only = _required(cur.fetchone())
            state.bounds.update(
                snapshot_at=_jsonable(snapshot_at), isolation=isolation, read_only=read_only
            )
            state.row_counts["legacy_s2s_results"] = int(row_count)

            cur.execute(
                """
                SELECT count(*)::bigint, count(DISTINCT o.id)::bigint, min(o.captured_at), max(o.captured_at),
                       count(DISTINCT o.run_id)::bigint
                FROM benchmarks_v2.benchmark_observations o
                WHERE o.benchmark = 'S2S'
                """
            )
            row_count, observation_count, min_created, max_created, run_count = _required(
                cur.fetchone()
            )
            state.bounds.update(
                {
                    "normalized_observation_created_at_min": _jsonable(min_created),
                    "normalized_observation_created_at_max": _jsonable(max_created),
                    "normalized_run_count": int(run_count),
                }
            )
            state.row_counts["normalized_s2s_observations"] = int(observation_count)

            cur.execute(
                """
                WITH legacy AS (
                  SELECT DISTINCT r.provider, r.benchmark,
                         NULLIF(split_part(r.audio_filename, '/', 1), '') AS external_run_id,
                         r.metric_type
                  FROM benchmarks_v2.results r
                  JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
                  WHERE r.benchmark = 'S2S' AND rn.status IN ('succeeded', 'partial')
                    AND NULLIF(BTRIM(r.audio_filename), '') IS NOT NULL
                    AND strpos(r.audio_filename, '/') > 0
                ), normalized AS (
                  SELECT DISTINCT o.provider, o.benchmark,
                         NULLIF(split_part(o.sample_id, '/', 1), '') AS external_run_id,
                         e.metric_type
                  FROM benchmarks_v2.benchmark_observations o
                  JOIN benchmarks_v2.metric_evaluations e ON e.observation_id = o.id
                  JOIN benchmarks_v2.runs rn ON rn.id = o.run_id
                  WHERE o.benchmark = 'S2S' AND rn.status IN ('succeeded', 'partial')
                    AND NULLIF(BTRIM(o.sample_id), '') IS NOT NULL
                    AND strpos(o.sample_id, '/') > 0
                )
                SELECT (SELECT count(*) FROM legacy), (SELECT count(*) FROM normalized)
                """
            )
            legacy_keys, normalized_keys = _required(cur.fetchone())
            state.key_counts = {
                "legacy_eligible_keys": int(legacy_keys),
                "normalized_eligible_keys": int(normalized_keys),
            }

            cur.execute(
                """
                WITH legacy AS (
                  SELECT r.provider, r.benchmark, split_part(r.audio_filename, '/', 1) AS external_run_id,
                         r.metric_type, count(*)::bigint AS n
                  FROM benchmarks_v2.results r JOIN benchmarks_v2.runs rn ON rn.id=r.run_id
                  WHERE r.benchmark='S2S' AND rn.status IN ('succeeded','partial')
                    AND NULLIF(BTRIM(r.audio_filename),'') IS NOT NULL AND strpos(r.audio_filename,'/') > 0
                  GROUP BY 1,2,3,4
                ), normalized AS (
                  SELECT o.provider, o.benchmark, split_part(o.sample_id, '/', 1) AS external_run_id,
                         e.metric_type, count(*)::bigint AS n
                  FROM benchmarks_v2.benchmark_observations o
                  JOIN benchmarks_v2.metric_evaluations e ON e.observation_id=o.id
                  JOIN benchmarks_v2.runs rn ON rn.id=o.run_id
                  WHERE o.benchmark='S2S' AND rn.status IN ('succeeded','partial')
                    AND NULLIF(BTRIM(o.sample_id),'') IS NOT NULL AND strpos(o.sample_id,'/') > 0
                  GROUP BY 1,2,3,4
                ), diff AS (
                  SELECT COALESCE(l.provider,n.provider) AS provider,
                         COALESCE(l.benchmark,n.benchmark) AS benchmark,
                         COALESCE(l.external_run_id,n.external_run_id) AS external_run_id,
                         COALESCE(l.metric_type,n.metric_type) AS metric_type,
                         COALESCE(l.n,0)::bigint AS legacy_count, COALESCE(n.n,0)::bigint AS normalized_count
                  FROM legacy l FULL OUTER JOIN normalized n USING (provider,benchmark,external_run_id,metric_type)
                  WHERE COALESCE(l.n,0) <> COALESCE(n.n,0)
                )
                SELECT count(*) FILTER (WHERE legacy_count > 0 AND normalized_count = 0),
                       count(*) FILTER (WHERE normalized_count > 0 AND legacy_count = 0),
                       count(*) FILTER (WHERE legacy_count > 0 AND normalized_count > 0),
                       COALESCE(sum(abs(legacy_count-normalized_count)),0)
                FROM diff
                """
            )
            missing, extra, overlapping, row_delta = _required(cur.fetchone())
            state.mismatches = {
                "legacy_missing_from_normalized_keys": int(missing),
                "normalized_extra_keys": int(extra),
                "same_key_count_mismatches": int(overlapping),
                "absolute_key_row_delta": int(row_delta),
                "samples": [],
            }
            cur.execute(
                """
                WITH legacy AS (
                  SELECT r.provider,r.benchmark,split_part(r.audio_filename,'/',1) external_run_id,r.metric_type,count(*)::bigint n
                  FROM benchmarks_v2.results r JOIN benchmarks_v2.runs rn ON rn.id=r.run_id
                  WHERE r.benchmark='S2S' AND rn.status IN ('succeeded','partial') AND NULLIF(BTRIM(r.audio_filename),'') IS NOT NULL AND strpos(r.audio_filename,'/')>0
                  GROUP BY 1,2,3,4
                ), normalized AS (
                  SELECT o.provider,o.benchmark,split_part(o.sample_id,'/',1) external_run_id,e.metric_type,count(*)::bigint n
                  FROM benchmarks_v2.benchmark_observations o JOIN benchmarks_v2.metric_evaluations e ON e.observation_id=o.id JOIN benchmarks_v2.runs rn ON rn.id=o.run_id
                  WHERE o.benchmark='S2S' AND rn.status IN ('succeeded','partial') AND NULLIF(BTRIM(o.sample_id),'') IS NOT NULL AND strpos(o.sample_id,'/')>0
                  GROUP BY 1,2,3,4
                )
                SELECT COALESCE(l.provider,n.provider),COALESCE(l.benchmark,n.benchmark),COALESCE(l.external_run_id,n.external_run_id),COALESCE(l.metric_type,n.metric_type),COALESCE(l.n,0),COALESCE(n.n,0)
                FROM legacy l FULL OUTER JOIN normalized n USING (provider,benchmark,external_run_id,metric_type)
                WHERE l.n IS NULL OR n.n IS NULL
                ORDER BY 1,2,3,4 LIMIT %s
                """,
                (sample_limit,),
            )
            state.mismatches["samples"] = [
                dict(
                    zip(
                        (
                            "provider",
                            "benchmark",
                            "external_run_id",
                            "metric_type",
                            "legacy_count",
                            "normalized_count",
                        ),
                        row,
                    )
                )
                for row in cur.fetchall()
            ]

            cur.execute(
                """
                SELECT
                  count(*) FILTER (WHERE rn.status IN ('succeeded','partial') AND (NULLIF(BTRIM(split_part(r.audio_filename,'/',1)),'') IS NULL OR strpos(r.audio_filename,'/')<2 OR split_part(r.audio_filename,'/',2)=''))::bigint,
                  count(*) FILTER (WHERE r.status <> 'success')::bigint,
                  count(*) FILTER (WHERE r.metric_value IS NULL)::bigint,
                  count(*)::bigint
                FROM benchmarks_v2.results r JOIN benchmarks_v2.runs rn ON rn.id=r.run_id
                WHERE r.benchmark='S2S'
                """
            )
            legacy_unusable, legacy_failed, legacy_null, legacy_total = _required(cur.fetchone())
            cur.execute(
                """
                SELECT count(*) FILTER (WHERE rn.status IN ('succeeded','partial') AND (NULLIF(BTRIM(split_part(o.sample_id,'/',1)),'') IS NULL OR strpos(o.sample_id,'/')<2 OR split_part(o.sample_id,'/',2)=''))::bigint,
                       count(*) FILTER (WHERE e.status <> 'succeeded')::bigint,
                       count(*) FILTER (WHERE v.metric_evaluation_id IS NULL OR v.value IS NULL)::bigint,
                       count(*)::bigint
                FROM benchmarks_v2.benchmark_observations o
                JOIN benchmarks_v2.metric_evaluations e ON e.observation_id=o.id
                JOIN benchmarks_v2.runs rn ON rn.id=o.run_id
                LEFT JOIN benchmarks_v2.metric_values v ON v.metric_evaluation_id=e.id AND v.value_role='primary'
                WHERE o.benchmark='S2S'
                """
            )
            normalized_unusable, normalized_failed, normalized_null, normalized_total = _required(
                cur.fetchone()
            )
            state.unusable_identities = {
                "legacy_null_empty_or_missing_prefix": int(legacy_unusable),
                "normalized_null_empty_or_missing_prefix": int(normalized_unusable),
            }
            state.failed_null_metric_coverage = {
                "legacy_rows": int(legacy_total),
                "legacy_failed_status_rows": int(legacy_failed),
                "legacy_null_value_rows": int(legacy_null),
                "normalized_evaluations": int(normalized_total),
                "normalized_failed_status_evaluations": int(normalized_failed),
                "normalized_missing_or_null_primary_value_evaluations": int(normalized_null),
            }

            if include_explain:
                cur.execute(
                    "SELECT o.provider, o.benchmark, split_part(o.sample_id,'/',1), e.metric_type FROM benchmarks_v2.benchmark_observations o JOIN benchmarks_v2.metric_evaluations e ON e.observation_id=o.id WHERE o.benchmark='S2S' ORDER BY o.id LIMIT 1"
                )
                example = cur.fetchone()
                if example:
                    cur.execute(
                        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT 1 FROM benchmarks_v2.benchmark_observations o JOIN benchmarks_v2.metric_evaluations e ON e.observation_id=o.id JOIN benchmarks_v2.runs rn ON rn.id=o.run_id WHERE o.provider=%s AND o.benchmark=%s AND split_part(o.sample_id,'/',1)=%s AND e.metric_type=%s AND rn.status IN ('succeeded','partial') LIMIT 1",
                        example,
                    )
                    state.explain = _required(cur.fetchone())[0]
    return S2SAuditReport(
        bounds=state.bounds,
        row_counts=state.row_counts,
        key_counts=state.key_counts,
        mismatches=state.mismatches,
        failed_null_metric_coverage=state.failed_null_metric_coverage,
        unusable_identities=state.unusable_identities,
        explain=state.explain,
        sample_limit=sample_limit,
    )


def audit_s2s_dsn(dsn: str, **kwargs: Any) -> S2SAuditReport:
    """Connect using only the supplied DSN, run :func:`audit_s2s`, then close."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        return audit_s2s(conn, **kwargs)


def report_json(report: S2SAuditReport) -> str:
    """Serialize a report for the standalone wrapper."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str)
