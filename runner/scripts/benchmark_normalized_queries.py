# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare legacy and normalized benchmark-storage query plans.

The script creates an isolated schema in a disposable PostgreSQL database, seeds
the legacy and normalized layouts with equivalent synthetic STT data, verifies
that each query pair returns the same result, and measures the warmed query plans
with ``EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)``.

It intentionally benchmarks the production query shapes rather than Python API
overhead. The schema is dropped on exit unless ``--keep-schema`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import tuple_row

_DEFAULT_URL = "postgresql://postgres:postgres@127.0.0.1:5432/benchmarks"
_METRICS_PER_OBSERVATION = 3
type PlanSample = tuple[float, int, int, tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class Workload:
    """A legacy query and its semantically equivalent normalized query."""

    name: str
    legacy_sql: str
    normalized_sql: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Measurement:
    """Summary of repeated PostgreSQL plan executions."""

    median_ms: float
    p95_ms: float
    min_ms: float
    median_buffer_hits: int
    median_buffer_reads: int
    plan_nodes: tuple[str, ...]
    indexes: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    """Measured legacy and normalized performance for one workload."""

    workload: str
    legacy: Measurement
    normalized: Measurement
    normalized_over_legacy: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("BENCHMARK_DATABASE_URL", _DEFAULT_URL),
        help="Disposable PostgreSQL URL (default: BENCHMARK_DATABASE_URL or Docker Compose DB).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=300_000,
        help="Legacy metric rows to seed; rounded down to a multiple of three.",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--models",
        type=int,
        default=12,
        help="Provider/model pairs represented in each hourly bucket.",
    )
    parser.add_argument("--buckets", type=int, default=720, help="Hourly buckets to seed.")
    parser.add_argument("--result-limit", type=int, default=1_000, help="Recent-result row limit.")
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown", dest="output_format"
    )
    parser.add_argument(
        "--keep-schema",
        action="store_true",
        help="Keep the generated query_bench_* schema for manual EXPLAIN inspection.",
    )
    parser.add_argument(
        "--candidate-indexes",
        action="store_true",
        help="Add candidate normalized-read indexes before measuring.",
    )
    args = parser.parse_args()
    for field in ("rows", "warmups", "iterations", "models", "buckets", "result_limit"):
        if getattr(args, field) < (0 if field == "warmups" else 1):
            parser.error(f"--{field} must be positive" if field != "warmups" else "--warmups >= 0")
    if args.rows < _METRICS_PER_OBSERVATION:
        parser.error(f"--rows must be at least {_METRICS_PER_OBSERVATION}")
    return args


def _create_schema(conn: psycopg.Connection[Any], schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        cur.execute("SET jit = off")
        cur.execute(
            """
            CREATE UNLOGGED TABLE runs (
                id BIGINT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                status TEXT NOT NULL,
                scheduled_at TIMESTAMPTZ NOT NULL
            );

            CREATE UNLOGGED TABLE legacy_results (
                id BIGINT PRIMARY KEY,
                run_id BIGINT NOT NULL REFERENCES runs(id),
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                voice TEXT,
                benchmark TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_value DOUBLE PRECISION,
                metric_units TEXT,
                audio_filename TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL
            );
            CREATE INDEX legacy_results_run_id_idx ON legacy_results (run_id);
            CREATE INDEX legacy_results_provider_model_idx
                ON legacy_results (provider, model);
            CREATE INDEX legacy_results_benchmark_created_at_idx
                ON legacy_results (benchmark, created_at DESC);

            CREATE UNLOGGED TABLE benchmark_observations (
                id UUID PRIMARY KEY,
                run_id BIGINT NOT NULL REFERENCES runs(id),
                dataset_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                voice TEXT,
                benchmark TEXT NOT NULL,
                captured_at TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL,
                UNIQUE NULLS NOT DISTINCT (run_id, sample_id, provider, model, voice)
            );

            CREATE UNLOGGED TABLE metric_evaluations (
                id UUID PRIMARY KEY,
                observation_id UUID NOT NULL REFERENCES benchmark_observations(id),
                metric_type TEXT NOT NULL,
                metric_version TEXT NOT NULL,
                evaluation_variant TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE (observation_id, metric_type, metric_version, evaluation_variant)
            );

            CREATE UNLOGGED TABLE metric_values (
                metric_evaluation_id UUID NOT NULL REFERENCES metric_evaluations(id),
                value_key TEXT NOT NULL,
                unit TEXT NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                value_role TEXT NOT NULL,
                PRIMARY KEY (metric_evaluation_id, value_key)
            );
            CREATE UNIQUE INDEX metric_values_one_primary
                ON metric_values (metric_evaluation_id) WHERE value_role = 'primary';

            CREATE UNLOGGED TABLE legacy_results_by_bucket (
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                benchmark TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                bucket_at TIMESTAMPTZ NOT NULL,
                min_value DOUBLE PRECISION NOT NULL,
                p25 DOUBLE PRECISION NOT NULL,
                p50 DOUBLE PRECISION NOT NULL,
                p75 DOUBLE PRECISION NOT NULL,
                max_value DOUBLE PRECISION NOT NULL,
                value_sum DOUBLE PRECISION NOT NULL,
                sample_count INTEGER NOT NULL,
                PRIMARY KEY (provider, model, benchmark, dataset_id, metric_type, bucket_at)
            );
            CREATE INDEX legacy_results_by_bucket_series_idx
                ON legacy_results_by_bucket (benchmark, dataset_id, bucket_at);

            CREATE UNLOGGED TABLE metric_values_by_bucket (
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                benchmark TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_version TEXT NOT NULL,
                evaluation_variant TEXT NOT NULL,
                value_key TEXT NOT NULL,
                unit TEXT NOT NULL,
                bucket_at TIMESTAMPTZ NOT NULL,
                min_value DOUBLE PRECISION NOT NULL,
                p25 DOUBLE PRECISION NOT NULL,
                p50 DOUBLE PRECISION NOT NULL,
                p75 DOUBLE PRECISION NOT NULL,
                max_value DOUBLE PRECISION NOT NULL,
                value_sum DOUBLE PRECISION NOT NULL,
                sample_count INTEGER NOT NULL,
                PRIMARY KEY (
                    provider, model, benchmark, dataset_id, metric_type, metric_version,
                    evaluation_variant, value_key, bucket_at
                )
            );
            CREATE INDEX metric_values_by_bucket_bucket_at
                ON metric_values_by_bucket (bucket_at);
            """
        )
    conn.commit()


def _seed(conn: psycopg.Connection[Any], rows: int, models: int, buckets: int) -> int:
    observation_count = rows // _METRICS_PER_OBSERVATION
    effective_rows = observation_count * _METRICS_PER_OBSERVATION
    run_count = models * buckets
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO runs (id, dataset_id, status, scheduled_at)
            SELECT run_id, 'stt-v2', 'succeeded',
                   date_trunc('hour', now()) - interval '1 hour'
                       - ((run_id - 1) / %(models)s) * interval '1 hour'
            FROM generate_series(1, %(runs)s) AS generated(run_id)
            """,
            {"models": models, "runs": run_count},
        )
        cur.execute(
            """
            INSERT INTO benchmark_observations
                (id, run_id, dataset_id, sample_id, provider, model, voice,
                 benchmark, captured_at, status)
            SELECT md5('observation-' || observation_number::text)::uuid,
                   ((observation_number - 1) %% %(runs)s) + 1,
                   'stt-v2', 'sample-' || observation_number,
                   'provider-' || (((observation_number - 1) %% %(models)s) + 1),
                   'model-' || (((observation_number - 1) %% %(models)s) + 1),
                   NULL, 'STT',
                   run.scheduled_at + observation_number * interval '1 microsecond',
                   'succeeded'
            FROM generate_series(1, %(observations)s) AS generated(observation_number)
            JOIN runs run ON run.id = ((observation_number - 1) %% %(runs)s) + 1
            """,
            {"models": models, "observations": observation_count, "runs": run_count},
        )
        metric_rows = """
            SELECT observation_number, metric.slot, metric.metric_type, metric.unit,
                   CASE metric.metric_type
                       WHEN 'WER' THEN (observation_number %% 100)::float8 / 1000
                       WHEN 'TTFT' THEN 100 + (observation_number %% 1000)::float8
                       ELSE 200 + (observation_number %% 1000)::float8
                   END AS metric_value
            FROM generate_series(1, %(observations)s) AS generated(observation_number)
            CROSS JOIN (VALUES
                (1, 'WER', 'ratio'),
                (2, 'TTFT', 'ms'),
                (3, 'TTFS', 'ms')
            ) AS metric(slot, metric_type, unit)
        """
        cur.execute(
            f"""
            INSERT INTO legacy_results
                (id, run_id, provider, model, voice, benchmark, metric_type,
                 metric_value, metric_units, audio_filename, created_at, status)
            SELECT (metric.observation_number - 1) * {_METRICS_PER_OBSERVATION} + metric.slot,
                   ((metric.observation_number - 1) %% %(runs)s) + 1,
                   'provider-' || (((metric.observation_number - 1) %% %(models)s) + 1),
                   'model-' || (((metric.observation_number - 1) %% %(models)s) + 1),
                   NULL, 'STT', metric.metric_type, metric.metric_value, metric.unit,
                   'sample-' || metric.observation_number,
                   run.scheduled_at
                       + metric.observation_number * interval '1 microsecond',
                   'success'
            FROM ({metric_rows}) AS metric
            JOIN runs run
              ON run.id = ((metric.observation_number - 1) %% %(runs)s) + 1
            """,  # noqa: S608 -- metric_rows is a module-owned SQL fragment.
            {"models": models, "observations": observation_count, "runs": run_count},
        )
        cur.execute(
            f"""
            INSERT INTO metric_evaluations
                (id, observation_id, metric_type, metric_version, evaluation_variant, status)
            SELECT md5('evaluation-'
                       || ((metric.observation_number - 1) * {_METRICS_PER_OBSERVATION}
                           + metric.slot)::text)::uuid,
                   md5('observation-' || metric.observation_number::text)::uuid,
                   metric.metric_type, 'v1', 'default', 'succeeded'
            FROM ({metric_rows}) AS metric
            """,  # noqa: S608 -- metric_rows is a module-owned SQL fragment.
            {"observations": observation_count},
        )
        cur.execute(
            f"""
            INSERT INTO metric_values
                (metric_evaluation_id, value_key, unit, value, value_role)
            SELECT md5('evaluation-'
                       || ((metric.observation_number - 1) * {_METRICS_PER_OBSERVATION}
                           + metric.slot)::text)::uuid,
                   'primary', metric.unit, metric.metric_value, 'primary'
            FROM ({metric_rows}) AS metric
            """,  # noqa: S608 -- metric_rows is a module-owned SQL fragment.
            {"observations": observation_count},
        )
        cur.execute(_legacy_rollup_insert())
        cur.execute(_normalized_rollup_insert())
        cur.execute(
            "ANALYZE runs, legacy_results, benchmark_observations, metric_evaluations, "
            "metric_values, legacy_results_by_bucket, metric_values_by_bucket"
        )
    conn.commit()
    return effective_rows


def _add_candidate_indexes(conn: psycopg.Connection[Any]) -> None:
    """Add exactly the normalized indexes proposed by the migration."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX benchmark_observations_recent_results_idx
                ON benchmark_observations
                    (benchmark, dataset_id, captured_at DESC, id DESC)
                WHERE status = 'succeeded';
            CREATE INDEX metric_values_by_bucket_series_idx
                ON metric_values_by_bucket
                    (benchmark, dataset_id, metric_version, evaluation_variant,
                     value_key, bucket_at);
            ANALYZE benchmark_observations, metric_values_by_bucket;
            """
        )
    conn.commit()


def _legacy_rollup_insert() -> str:
    return """
        INSERT INTO legacy_results_by_bucket
            (provider, model, benchmark, dataset_id, metric_type, bucket_at,
             min_value, p25, p50, p75, max_value, value_sum, sample_count)
        SELECT result.provider, result.model, result.benchmark,
               COALESCE(run.dataset_id, '__all__'), result.metric_type,
               run.scheduled_at,
               MIN(result.metric_value)::float8,
               PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY result.metric_value)::float8,
               PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY result.metric_value)::float8,
               PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY result.metric_value)::float8,
               MAX(result.metric_value)::float8, SUM(result.metric_value)::float8, COUNT(*)::int
        FROM legacy_results result
        JOIN runs run ON run.id = result.run_id
        WHERE result.status = 'success' AND run.status IN ('succeeded', 'partial')
          AND result.metric_value IS NOT NULL
        GROUP BY GROUPING SETS (
            (result.provider, result.model, result.benchmark, run.dataset_id,
             result.metric_type, run.scheduled_at),
            (result.provider, result.model, result.benchmark,
             result.metric_type, run.scheduled_at)
        )
    """


def _normalized_rollup_insert() -> str:
    return """
        INSERT INTO metric_values_by_bucket
            (provider, model, benchmark, dataset_id, metric_type, metric_version,
             evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50,
             p75, max_value, value_sum, sample_count)
        SELECT observation.provider, observation.model, observation.benchmark,
               COALESCE(observation.dataset_id, '__all__'), evaluation.metric_type,
               evaluation.metric_version, evaluation.evaluation_variant, value.value_key,
               value.unit, run.scheduled_at,
               MIN(value.value)::float8,
               PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY value.value)::float8,
               PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY value.value)::float8,
               PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY value.value)::float8,
               MAX(value.value)::float8, SUM(value.value)::float8, COUNT(*)::int
        FROM metric_values value
        JOIN metric_evaluations evaluation
          ON evaluation.id = value.metric_evaluation_id
        JOIN benchmark_observations observation
          ON observation.id = evaluation.observation_id
        JOIN runs run ON run.id = observation.run_id
        WHERE evaluation.status = 'succeeded' AND run.status IN ('succeeded', 'partial')
        GROUP BY GROUPING SETS (
            (observation.provider, observation.model, observation.benchmark,
             observation.dataset_id, evaluation.metric_type, evaluation.metric_version,
             evaluation.evaluation_variant, value.value_key, value.unit, run.scheduled_at),
            (observation.provider, observation.model, observation.benchmark,
             evaluation.metric_type, evaluation.metric_version,
             evaluation.evaluation_variant, value.value_key, value.unit, run.scheduled_at)
        )
    """


def _workloads(result_limit: int) -> tuple[Workload, ...]:
    recent_legacy = """
        SELECT result.run_id, result.provider, result.model, result.voice, result.benchmark,
               run.dataset_id, result.metric_type, result.metric_value, result.metric_units,
               result.audio_filename AS sample_id, result.created_at,
               UPPER(run.status) AS run_status
        FROM legacy_results result
        JOIN runs run ON run.id = result.run_id
        WHERE result.status = 'success'
          AND run.status IN ('succeeded', 'partial')
          AND result.benchmark = %(benchmark)s
          AND result.metric_type = %(metric_type)s
          AND run.dataset_id = %(dataset_id)s
          AND result.created_at >= NOW() - INTERVAL '7 days'
        ORDER BY result.created_at DESC, result.id DESC
        LIMIT %(limit)s
    """
    recent_normalized = """
        SELECT observation.run_id, observation.provider, observation.model, observation.voice,
               observation.benchmark, observation.dataset_id, evaluation.metric_type,
               value.value, value.unit, observation.sample_id, observation.captured_at,
               UPPER(run.status) AS run_status
        FROM benchmark_observations observation
        JOIN runs run ON run.id = observation.run_id
        JOIN metric_evaluations evaluation
          ON evaluation.observation_id = observation.id
        JOIN metric_values value
          ON value.metric_evaluation_id = evaluation.id AND value.value_role = 'primary'
        WHERE observation.status = 'succeeded'
          AND evaluation.status = 'succeeded'
          AND run.status IN ('succeeded', 'partial')
          AND observation.benchmark = %(benchmark)s
          AND evaluation.metric_type = %(metric_type)s
          AND evaluation.metric_version = 'v1'
          AND evaluation.evaluation_variant = 'default'
          AND observation.dataset_id = %(dataset_id)s
          AND observation.captured_at >= NOW() - INTERVAL '7 days'
        ORDER BY observation.captured_at DESC, observation.id DESC
        LIMIT %(limit)s
    """
    series_legacy = """
        SELECT provider, model, metric_type, bucket_at,
               min_value, p25, p50, p75, max_value, value_sum, sample_count
        FROM legacy_results_by_bucket
        WHERE benchmark = %(benchmark)s
          AND dataset_id = %(dataset_id)s
          AND bucket_at >= NOW() - INTERVAL '7 days'
        ORDER BY bucket_at, provider, model, metric_type
    """
    series_normalized = """
        SELECT provider, model, metric_type, bucket_at,
               min_value, p25, p50, p75, max_value, value_sum, sample_count
        FROM metric_values_by_bucket
        WHERE benchmark = %(benchmark)s
          AND dataset_id = %(dataset_id)s
          AND metric_version = 'v1'
          AND evaluation_variant = 'default'
          AND value_key = 'primary'
          AND bucket_at >= NOW() - INTERVAL '7 days'
        ORDER BY bucket_at, provider, model, metric_type
    """
    common = {"benchmark": "STT", "metric_type": "WER", "dataset_id": "stt-v2"}
    return (
        Workload(
            "recent_results",
            recent_legacy,
            recent_normalized,
            {**common, "limit": result_limit},
        ),
        Workload("dashboard_series", series_legacy, series_normalized, common),
    )


def _verify_parity(conn: psycopg.Connection[Any], workload: Workload) -> None:
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(workload.legacy_sql, workload.params)
        legacy_rows = cur.fetchall()
        cur.execute(workload.normalized_sql, workload.params)
        normalized_rows = cur.fetchall()
    equal = len(legacy_rows) == len(normalized_rows) and all(
        len(legacy) == len(normalized)
        and all(
            math.isclose(legacy_value, normalized_value, rel_tol=1e-12, abs_tol=1e-12)
            if isinstance(legacy_value, float) and isinstance(normalized_value, float)
            else legacy_value == normalized_value
            for legacy_value, normalized_value in zip(legacy, normalized, strict=True)
        )
        for legacy, normalized in zip(legacy_rows, normalized_rows, strict=True)
    )
    if not equal:
        raise RuntimeError(
            f"{workload.name} results differ: legacy={len(legacy_rows)}, "
            f"normalized={len(normalized_rows)}"
        )


def _walk_plan(node: dict[str, Any]) -> tuple[list[str], list[str]]:
    nodes = [str(node["Node Type"])]
    indexes = [str(node["Index Name"])] if "Index Name" in node else []
    for child in node.get("Plans", []):
        child_nodes, child_indexes = _walk_plan(child)
        nodes.extend(child_nodes)
        indexes.extend(child_indexes)
    return nodes, indexes


def _buffer_totals(plan: dict[str, Any]) -> tuple[int, int]:
    hits = int(plan.get("Shared Hit Blocks", 0)) + int(plan.get("Local Hit Blocks", 0))
    reads = int(plan.get("Shared Read Blocks", 0)) + int(plan.get("Local Read Blocks", 0))
    return hits, reads


def _explain(conn: psycopg.Connection[Any], query: str, params: dict[str, Any]) -> PlanSample:
    statement = sql.SQL("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ") + sql.SQL(query)
    with conn.cursor() as cur:
        # Prevent psycopg from switching plan preparation partway through the samples.
        cur.execute(statement, params, prepare=False)
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("EXPLAIN returned no plan")
    document = row[0][0]
    plan = document["Plan"]
    nodes, indexes = _walk_plan(plan)
    hits, reads = _buffer_totals(plan)
    return (
        float(document["Execution Time"]),
        hits,
        reads,
        tuple(dict.fromkeys(nodes)),
        tuple(dict.fromkeys(indexes)),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _summarize(samples: Sequence[PlanSample]) -> Measurement:
    durations = [sample[0] for sample in samples]
    hits = [sample[1] for sample in samples]
    reads = [sample[2] for sample in samples]
    return Measurement(
        median_ms=statistics.median(durations),
        p95_ms=_percentile(durations, 0.95),
        min_ms=min(durations),
        median_buffer_hits=int(statistics.median(hits)),
        median_buffer_reads=int(statistics.median(reads)),
        plan_nodes=samples[-1][3],
        indexes=samples[-1][4],
    )


def _measure(
    conn: psycopg.Connection[Any], workload: Workload, warmups: int, iterations: int
) -> Comparison:
    for _ in range(warmups):
        _explain(conn, workload.legacy_sql, workload.params)
        _explain(conn, workload.normalized_sql, workload.params)
    legacy_samples: list[PlanSample] = []
    normalized_samples: list[PlanSample] = []
    for iteration in range(iterations):
        pair = (
            ((workload.legacy_sql, legacy_samples), (workload.normalized_sql, normalized_samples))
            if iteration % 2 == 0
            else (
                (workload.normalized_sql, normalized_samples),
                (workload.legacy_sql, legacy_samples),
            )
        )
        for query, samples in pair:
            samples.append(_explain(conn, query, workload.params))
    legacy = _summarize(legacy_samples)
    normalized = _summarize(normalized_samples)
    return Comparison(
        workload=workload.name,
        legacy=legacy,
        normalized=normalized,
        normalized_over_legacy=normalized.median_ms / legacy.median_ms,
    )


def _markdown(
    comparisons: Sequence[Comparison],
    *,
    rows: int,
    schema: str,
    postgres_version: str,
    index_profile: str,
    result_limit: int,
) -> str:
    lines = [
        f"Seeded rows: {rows:,}",
        f"PostgreSQL: {postgres_version}",
        f"Schema: {schema}",
        f"Normalized indexes: {index_profile}",
        f"Recent-results limit: {result_limit:,}",
        "",
        "| Workload | Legacy median | Normalized median | Normalized / legacy | "
        "Legacy p95 | Normalized p95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        lines.append(
            f"| {comparison.workload} | {comparison.legacy.median_ms:.3f} ms | "
            f"{comparison.normalized.median_ms:.3f} ms | "
            f"{comparison.normalized_over_legacy:.2f}x | "
            f"{comparison.legacy.p95_ms:.3f} ms | "
            f"{comparison.normalized.p95_ms:.3f} ms |"
        )
    lines.extend(("", "Plan details:"))
    for comparison in comparisons:
        lines.extend(
            (
                f"- {comparison.workload} legacy: nodes={','.join(comparison.legacy.plan_nodes)}; "
                f"indexes={','.join(comparison.legacy.indexes) or 'none'}; "
                f"buffers hit/read={comparison.legacy.median_buffer_hits}/"
                f"{comparison.legacy.median_buffer_reads}",
                f"- {comparison.workload} normalized: "
                f"nodes={','.join(comparison.normalized.plan_nodes)}; "
                f"indexes={','.join(comparison.normalized.indexes) or 'none'}; "
                f"buffers hit/read={comparison.normalized.median_buffer_hits}/"
                f"{comparison.normalized.median_buffer_reads}",
            )
        )
    return "\n".join(lines)


def main() -> None:
    """Seed equivalent schemas, verify query parity, and print plan timings."""
    args = _parse_args()
    schema = f"query_bench_{uuid4().hex[:12]}"
    effective_rows = args.rows - args.rows % _METRICS_PER_OBSERVATION
    with psycopg.connect(args.database_url, autocommit=False) as conn:
        try:
            _create_schema(conn, schema)
            effective_rows = _seed(conn, effective_rows, args.models, args.buckets)
            if args.candidate_indexes:
                _add_candidate_indexes(conn)
            workloads = _workloads(args.result_limit)
            for workload in workloads:
                _verify_parity(conn, workload)
            comparisons = [
                _measure(conn, workload, args.warmups, args.iterations) for workload in workloads
            ]
            with conn.cursor() as cur:
                cur.execute("SHOW server_version")
                version_row = cur.fetchone()
            postgres_version = str(version_row[0]) if version_row is not None else "unknown"
            if args.output_format == "json":
                print(
                    json.dumps(
                        {
                            "rows": effective_rows,
                            "schema": schema,
                            "postgres_version": postgres_version,
                            "normalized_indexes": (
                                "candidate" if args.candidate_indexes else "current"
                            ),
                            "result_limit": args.result_limit,
                            "comparisons": [asdict(comparison) for comparison in comparisons],
                        },
                        indent=2,
                    )
                )
            else:
                print(
                    _markdown(
                        comparisons,
                        rows=effective_rows,
                        schema=schema,
                        postgres_version=postgres_version,
                        index_profile="candidate" if args.candidate_indexes else "current",
                        result_limit=args.result_limit,
                    )
                )
        finally:
            if args.keep_schema:
                print(f"Kept schema: {schema}", file=sys.stderr)
            else:
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                    )
                conn.commit()


if __name__ == "__main__":
    main()
