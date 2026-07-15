# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the dataset dimension to the stats matviews and the series rollup.

The per-window matviews (results_24h/7d/30d) and results_by_bucket grouped by
(provider, model, benchmark, metric_type) only, pooling every dataset into one
row. Both now carry ``dataset_id`` and materialize two granularities via
GROUPING SETS: one row per dataset, plus a pooled row under the ``'__all__'``
sentinel that reproduces the previous numbers exactly (pooled percentiles are
not derivable from per-dataset percentiles, so both must be materialized).

Dataset attribution joins through runs, with TTS rows pinned to ``tts-v1``:
a 'both' run's row records only the STT dataset id, and historical TTS-only
runs recorded the env default.

The sentinel is a literal (not NULL) so each matview keeps a plain-column
unique index, which REFRESH MATERIALIZED VIEW CONCURRENTLY requires.

results_by_bucket is dropped and rebuilt from raw results with ``dataset_id``
in the primary key. Migrations deploy separately from images, so rerun the
backfill (truncate + the ``upgrade`` insert-select) once the new image is
live. Same recovery if the table ever drifts from ``results``.
"""

from __future__ import annotations

from alembic import op

revision = "20260715_0010"
down_revision = "20260707_0009"
branch_labels = None
depends_on = None

_WINDOWS: dict[str, str] = {
    "results_24h": "24 hours",
    "results_7d": "7 days",
    "results_30d": "30 days",
}

_DATASET_CASE = "CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1' ELSE rn.dataset_id END"

# Settings.schedule_period_seconds default; only used for legacy rows with a
# null scheduled_at.
_SCHEDULE_PERIOD_SECONDS = 1800

_BUCKET_EXPR = f"""COALESCE(
    rn.scheduled_at,
    to_timestamp(
        floor(extract(epoch FROM r.created_at) / {_SCHEDULE_PERIOD_SECONDS})
        * {_SCHEDULE_PERIOD_SECONDS}
    )
)"""


def _view_sql(name: str, interval: str) -> str:
    # S608 false-positive: interpolations come from module constants.
    return f"""
        CREATE MATERIALIZED VIEW benchmarks_v2.{name} AS
        SELECT provider, model, benchmark, dataset_id, metric_type,
               avg_value, stddev_value, min_value,
               pct[1] AS p25, pct[2] AS p50, pct[3] AS p75,
               pct[4] AS p90, pct[5] AS p95, pct[6] AS p99,
               max_value, sample_count
        FROM (
            SELECT r.provider, r.model, r.benchmark,
                   COALESCE({_DATASET_CASE}, '__all__') AS dataset_id,
                   r.metric_type,
                   AVG(r.metric_value)::float8 AS avg_value,
                   COALESCE(STDDEV_SAMP(r.metric_value), 0)::float8 AS stddev_value,
                   MIN(r.metric_value)::float8 AS min_value,
                   PERCENTILE_CONT(ARRAY[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
                       WITHIN GROUP (ORDER BY r.metric_value)::float8[] AS pct,
                   MAX(r.metric_value)::float8 AS max_value,
                   COUNT(*)::int AS sample_count
            FROM benchmarks_v2.results r
            JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
            WHERE r.status = 'success'
              AND rn.status IN ('succeeded', 'partial')
              AND r.metric_value IS NOT NULL
              AND r.created_at >= now() - INTERVAL '{interval}'
            GROUP BY GROUPING SETS (
                (r.provider, r.model, r.benchmark, r.metric_type, {_DATASET_CASE}),
                (r.provider, r.model, r.benchmark, r.metric_type)
            )
        ) stats
    """  # noqa: S608


_BUCKET_TABLE_SQL = """
    CREATE TABLE benchmarks_v2.results_by_bucket (
        provider      TEXT NOT NULL,
        model         TEXT NOT NULL,
        benchmark     TEXT NOT NULL CHECK (benchmark IN ('STT','TTS','S2S')),
        dataset_id    TEXT NOT NULL,
        metric_type   TEXT NOT NULL,
        bucket_at     TIMESTAMPTZ NOT NULL,
        min_value     DOUBLE PRECISION NOT NULL,
        p25           DOUBLE PRECISION NOT NULL,
        p50           DOUBLE PRECISION NOT NULL,
        p75           DOUBLE PRECISION NOT NULL,
        max_value     DOUBLE PRECISION NOT NULL,
        value_sum     DOUBLE PRECISION NOT NULL,
        sample_count  INTEGER NOT NULL,
        PRIMARY KEY (provider, model, benchmark, dataset_id, metric_type, bucket_at)
    )
"""

# S608 false-positive: interpolations come from module constants.
_BACKFILL_SQL = f"""
    INSERT INTO benchmarks_v2.results_by_bucket
        (provider, model, benchmark, dataset_id, metric_type, bucket_at,
         min_value, p25, p50, p75, max_value, value_sum, sample_count)
    SELECT r.provider, r.model, r.benchmark,
           COALESCE({_DATASET_CASE}, '__all__'),
           r.metric_type,
           {_BUCKET_EXPR},
           MIN(r.metric_value)::float8,
           PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY r.metric_value)::float8,
           PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY r.metric_value)::float8,
           PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.metric_value)::float8,
           MAX(r.metric_value)::float8,
           SUM(r.metric_value)::float8,
           COUNT(*)::int
    FROM benchmarks_v2.results r
    JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
    WHERE r.status = 'success'
      AND rn.status IN ('succeeded', 'partial')
      AND r.metric_value IS NOT NULL
    GROUP BY GROUPING SETS (
        (r.provider, r.model, r.benchmark, r.metric_type, {_DATASET_CASE}, {_BUCKET_EXPR}),
        (r.provider, r.model, r.benchmark, r.metric_type, {_BUCKET_EXPR})
    )
"""  # noqa: S608


def upgrade() -> None:
    """Recreate the matviews and the rollup table with the dataset dimension."""
    for name, interval in _WINDOWS.items():
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.{name}")
        op.execute(_view_sql(name, interval))
        op.execute(
            f"CREATE UNIQUE INDEX {name}_group_key "
            f"ON benchmarks_v2.{name} (provider, model, benchmark, dataset_id, metric_type)"
        )
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.results_by_bucket")
    op.execute(_BUCKET_TABLE_SQL)
    op.execute(
        "CREATE INDEX results_by_bucket_series_idx "
        "ON benchmarks_v2.results_by_bucket (benchmark, bucket_at)"
    )
    op.execute(_BACKFILL_SQL)


def downgrade() -> None:
    """Restore the pooled-only matviews and rollup table (shapes of 0005/0006/0009)."""
    for name, interval in _WINDOWS.items():
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.{name}")
        op.execute(f"""
            CREATE MATERIALIZED VIEW benchmarks_v2.{name} AS
            SELECT provider, model, benchmark, metric_type,
                   avg_value, stddev_value, min_value,
                   pct[1] AS p25, pct[2] AS p50, pct[3] AS p75,
                   pct[4] AS p90, pct[5] AS p95, pct[6] AS p99,
                   max_value, sample_count
            FROM (
                SELECT r.provider, r.model, r.benchmark, r.metric_type,
                       AVG(r.metric_value)::float8 AS avg_value,
                       COALESCE(STDDEV_SAMP(r.metric_value), 0)::float8 AS stddev_value,
                       MIN(r.metric_value)::float8 AS min_value,
                       PERCENTILE_CONT(ARRAY[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
                           WITHIN GROUP (ORDER BY r.metric_value)::float8[] AS pct,
                       MAX(r.metric_value)::float8 AS max_value,
                       COUNT(*)::int AS sample_count
                FROM benchmarks_v2.results r
                JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
                WHERE r.status = 'success'
                  AND rn.status IN ('succeeded', 'partial')
                  AND r.metric_value IS NOT NULL
                  AND r.created_at >= now() - INTERVAL '{interval}'
                GROUP BY r.provider, r.model, r.benchmark, r.metric_type
            ) stats
        """)  # noqa: S608
        op.execute(
            f"CREATE UNIQUE INDEX {name}_group_key "
            f"ON benchmarks_v2.{name} (provider, model, benchmark, metric_type)"
        )
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.results_by_bucket")
    op.execute("""
        CREATE TABLE benchmarks_v2.results_by_bucket (
            provider      TEXT NOT NULL,
            model         TEXT NOT NULL,
            benchmark     TEXT NOT NULL CHECK (benchmark IN ('STT','TTS','S2S')),
            metric_type   TEXT NOT NULL,
            bucket_at     TIMESTAMPTZ NOT NULL,
            min_value     DOUBLE PRECISION NOT NULL,
            p25           DOUBLE PRECISION NOT NULL,
            p50           DOUBLE PRECISION NOT NULL,
            p75           DOUBLE PRECISION NOT NULL,
            max_value     DOUBLE PRECISION NOT NULL,
            value_sum     DOUBLE PRECISION NOT NULL,
            sample_count  INTEGER NOT NULL,
            PRIMARY KEY (provider, model, benchmark, metric_type, bucket_at)
        )
    """)
    op.execute(
        "CREATE INDEX results_by_bucket_series_idx "
        "ON benchmarks_v2.results_by_bucket (benchmark, bucket_at)"
    )
    op.execute(f"""
        INSERT INTO benchmarks_v2.results_by_bucket
            (provider, model, benchmark, metric_type, bucket_at,
             min_value, p25, p50, p75, max_value, value_sum, sample_count)
        SELECT r.provider, r.model, r.benchmark, r.metric_type, {_BUCKET_EXPR},
               MIN(r.metric_value)::float8,
               PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY r.metric_value)::float8,
               PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY r.metric_value)::float8,
               PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.metric_value)::float8,
               MAX(r.metric_value)::float8,
               SUM(r.metric_value)::float8,
               COUNT(*)::int
        FROM benchmarks_v2.results r
        JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
        WHERE r.status = 'success'
          AND rn.status IN ('succeeded', 'partial')
          AND r.metric_value IS NOT NULL
        GROUP BY r.provider, r.model, r.benchmark, r.metric_type, {_BUCKET_EXPR}
    """)  # noqa: S608
