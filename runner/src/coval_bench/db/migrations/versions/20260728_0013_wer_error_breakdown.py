# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry the WER error-type breakdown through to the stats matviews.

Stored as percentage points of ``metric_value`` (not raw counts) so AVG of the
three parts sums to ``avg_value`` exactly — the dashboard's split must
reconcile to the total it sits under. Nullable, no backfill: pre-migration rows
keep NULL and clients fall back to the plain total. The matviews are dropped
and recreated (a matview's SELECT cannot be altered); otherwise unchanged from
migration 0010.
"""

from __future__ import annotations

from alembic import op

revision = "20260728_0013"
down_revision = "20260727_0012"
branch_labels = None
depends_on = None

_COLUMNS: tuple[str, ...] = (
    "wer_insertions_pct",
    "wer_deletions_pct",
    "wer_substitutions_pct",
)

_WINDOWS: dict[str, str] = {
    "results_24h": "24 hours",
    "results_7d": "7 days",
    "results_30d": "30 days",
}

_DATASET_CASE = "CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1' ELSE rn.dataset_id END"


def _view_sql(name: str, interval: str, *, breakdown: bool) -> str:
    """Render the per-window matview; ``breakdown=False`` is 0010's shape for downgrade."""
    outer = "".join(f", {c}" for c in _COLUMNS) if breakdown else ""
    # COUNT guard: AVG over a scored/legacy mix wouldn't reconcile with
    # avg_value, so a partial group reports no breakdown at all.
    inner = (
        "".join(
            f", CASE WHEN COUNT(r.{c}) = COUNT(*) THEN AVG(r.{c})::float8 END AS {c}"
            for c in _COLUMNS
        )
        if breakdown
        else ""
    )
    # S608 false-positive: interpolations come from module constants.
    return f"""
        CREATE MATERIALIZED VIEW benchmarks_v2.{name} AS
        SELECT provider, model, benchmark, dataset_id, metric_type,
               avg_value, stddev_value, min_value,
               pct[1] AS p25, pct[2] AS p50, pct[3] AS p75,
               pct[4] AS p90, pct[5] AS p95, pct[6] AS p99,
               max_value, sample_count{outer}
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
                   COUNT(*)::int AS sample_count{inner}
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


def _rebuild_views(*, breakdown: bool) -> None:
    for name, interval in _WINDOWS.items():
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.{name}")
        op.execute(_view_sql(name, interval, breakdown=breakdown))
        op.execute(
            f"CREATE UNIQUE INDEX {name}_group_key "
            f"ON benchmarks_v2.{name} (provider, model, benchmark, dataset_id, metric_type)"
        )


def upgrade() -> None:
    for column in _COLUMNS:
        op.execute(f"ALTER TABLE benchmarks_v2.results ADD COLUMN {column} DOUBLE PRECISION")
    _rebuild_views(breakdown=True)


def downgrade() -> None:
    _rebuild_views(breakdown=False)
    for column in _COLUMNS:
        op.execute(f"ALTER TABLE benchmarks_v2.results DROP COLUMN {column}")
