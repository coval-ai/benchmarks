# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add a dataset dimension to the per-window matviews and results_by_bucket.

Multi-dataset benchmarking runs each dataset as its own run (one DATASET_ID
per execution), so a result's dataset comes from its parent run. Both
aggregation layers gain a ``dataset_id`` group key derived via the run join:

    CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1' ELSE rn.dataset_id END

The CASE covers the one attribution gap: a run row records the STT dataset id,
while TTS results in the same run always come from the fixed ``tts-v1``
manifest (hardcoded in the orchestrator). S2S runs record ``s2s-v1``
themselves. ``RunWriter.refresh_bucket`` computes the same expression and must
stay in sync.

``results_by_bucket.dataset_id`` carries ``DEFAULT 'unknown'`` so a runner
image predating this migration can still refresh buckets during the
migration→image-bump gap. Migrations deploy separately from images: once the
new image is live, rerun the backfill (truncate + the ``upgrade``
insert-select) to heal any 'unknown' rows — same recovery as 20260611_0006.

Until runs with different STT dataset ids exist inside a window, every group
holds exactly one dataset and row counts are unchanged, so pre-existing API
readers see identical responses through the deploy window.
"""

from __future__ import annotations

from alembic import op

revision = "20260713_0010"
down_revision = "20260707_0009"
branch_labels = None
depends_on = None

_WINDOWS: dict[str, str] = {
    "results_24h": "24 hours",
    "results_7d": "7 days",
    "results_30d": "30 days",
}

_DATASET_SQL = "CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1' ELSE rn.dataset_id END"

# Settings.schedule_period_seconds default; only used for legacy rows with a
# null scheduled_at.
_SCHEDULE_PERIOD_SECONDS = 1800

_BUCKET_AT_SQL = f"""COALESCE(
    rn.scheduled_at,
    to_timestamp(
        floor(extract(epoch FROM r.created_at) / {_SCHEDULE_PERIOD_SECONDS})
        * {_SCHEDULE_PERIOD_SECONDS}
    )
)"""


def _view_sql(name: str, interval: str, *, with_dataset: bool) -> str:
    # S608 false-positive: every interpolated fragment is a module constant.
    dataset_col = "dataset_id, " if with_dataset else ""
    dataset_expr = f"{_DATASET_SQL} AS dataset_id," if with_dataset else ""
    dataset_group = f", {_DATASET_SQL}" if with_dataset else ""
    return f"""
        CREATE MATERIALIZED VIEW benchmarks_v2.{name} AS
        SELECT provider, model, benchmark, {dataset_col}metric_type,
               avg_value, stddev_value, min_value,
               pct[1] AS p25, pct[2] AS p50, pct[3] AS p75,
               pct[4] AS p90, pct[5] AS p95, pct[6] AS p99,
               max_value, sample_count
        FROM (
            SELECT r.provider, r.model, r.benchmark, {dataset_expr}
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
            GROUP BY r.provider, r.model, r.benchmark{dataset_group}, r.metric_type
        ) stats
    """  # noqa: S608


def _bucket_backfill_sql(*, with_dataset: bool) -> str:
    # S608 false-positive: every interpolated fragment is a module constant.
    dataset_col = "dataset_id, " if with_dataset else ""
    dataset_expr = f"{_DATASET_SQL}, " if with_dataset else ""
    dataset_group = f", {_DATASET_SQL}" if with_dataset else ""
    return f"""
        INSERT INTO benchmarks_v2.results_by_bucket
            (provider, model, benchmark, {dataset_col}metric_type, bucket_at,
             min_value, p25, p50, p75, max_value, value_sum, sample_count)
        SELECT provider, model, benchmark, {dataset_col}metric_type, bucket_at,
               min_value, pct[1], pct[2], pct[3], max_value, value_sum, sample_count
        FROM (
            SELECT r.provider, r.model, r.benchmark, {dataset_expr}r.metric_type,
                   {_BUCKET_AT_SQL} AS bucket_at,
                   MIN(r.metric_value)::float8 AS min_value,
                   (PERCENTILE_CONT(ARRAY[0.25, 0.5, 0.75])
                       WITHIN GROUP (ORDER BY r.metric_value))::float8[] AS pct,
                   MAX(r.metric_value)::float8 AS max_value,
                   SUM(r.metric_value)::float8 AS value_sum,
                   COUNT(*)::int AS sample_count
            FROM benchmarks_v2.results r
            JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
            WHERE r.status = 'success'
              AND rn.status IN ('succeeded', 'partial')
              AND r.metric_value IS NOT NULL
            GROUP BY r.provider, r.model, r.benchmark{dataset_group}, r.metric_type,
                     {_BUCKET_AT_SQL}
        ) agg
    """  # noqa: S608


def _bucket_table_sql(*, with_dataset: bool) -> str:
    dataset_col = (
        "dataset_id    TEXT NOT NULL DEFAULT 'unknown',\n            " if with_dataset else ""
    )
    dataset_key = "dataset_id, " if with_dataset else ""
    return f"""
        CREATE TABLE benchmarks_v2.results_by_bucket (
            provider      TEXT NOT NULL,
            model         TEXT NOT NULL,
            benchmark     TEXT NOT NULL CHECK (benchmark IN ('STT','TTS','S2S')),
            {dataset_col}metric_type   TEXT NOT NULL,
            bucket_at     TIMESTAMPTZ NOT NULL,
            min_value     DOUBLE PRECISION NOT NULL,
            p25           DOUBLE PRECISION NOT NULL,
            p50           DOUBLE PRECISION NOT NULL,
            p75           DOUBLE PRECISION NOT NULL,
            max_value     DOUBLE PRECISION NOT NULL,
            value_sum     DOUBLE PRECISION NOT NULL,
            sample_count  INTEGER NOT NULL,
            PRIMARY KEY (provider, model, benchmark, {dataset_key}metric_type, bucket_at)
        )
    """


def _rebuild(*, with_dataset: bool) -> None:
    for name, interval in _WINDOWS.items():
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.{name}")
        op.execute(_view_sql(name, interval, with_dataset=with_dataset))
        group_key = (
            "(provider, model, benchmark, dataset_id, metric_type)"
            if with_dataset
            else "(provider, model, benchmark, metric_type)"
        )
        op.execute(f"CREATE UNIQUE INDEX {name}_group_key ON benchmarks_v2.{name} {group_key}")
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.results_by_bucket")
    op.execute(_bucket_table_sql(with_dataset=with_dataset))
    op.execute(
        "CREATE INDEX results_by_bucket_series_idx "
        "ON benchmarks_v2.results_by_bucket (benchmark, bucket_at)"
    )
    op.execute(_bucket_backfill_sql(with_dataset=with_dataset))


def upgrade() -> None:
    """Rebuild both aggregation layers with the dataset_id group key."""
    _rebuild(with_dataset=True)


def downgrade() -> None:
    """Restore the pre-dataset shapes (20260611_0005/0006 + the 0009 CHECK)."""
    _rebuild(with_dataset=False)
