# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Series rollup table: results_by_bucket.

One row per (provider, model, benchmark, metric_type, bucket_at):
min/p25/p50/p75/max/value_sum/sample_count. ``bucket_at`` is the run's cron
trigger time, falling back to ``created_at`` floored to the scheduler period
for legacy rows. The aggregates series block reads this table; the
orchestrator's end-of-run hook (``RunWriter.refresh_bucket``) keeps it
current. This migration creates the table and backfills it from existing
results.

Migrations deploy separately from images, so rerun the backfill (truncate +
the ``upgrade`` insert-select) once the new image is live. Same recovery if
the table ever drifts from ``results``.
"""

from __future__ import annotations

from alembic import op

revision = "20260611_0006"
down_revision = "20260611_0005"
branch_labels = None
depends_on = None

# Settings.schedule_period_seconds default; only used for legacy rows with a
# null scheduled_at.
_SCHEDULE_PERIOD_SECONDS = 1800

# S608 false-positive: only the integer constant above is interpolated.
_BACKFILL_SQL = f"""
    INSERT INTO benchmarks_v2.results_by_bucket
        (provider, model, benchmark, metric_type, bucket_at,
         min_value, p25, p50, p75, max_value, value_sum, sample_count)
    SELECT provider, model, benchmark, metric_type, bucket_at,
           min_value, pct[1], pct[2], pct[3], max_value, value_sum, sample_count
    FROM (
        SELECT r.provider, r.model, r.benchmark, r.metric_type,
               COALESCE(
                   rn.scheduled_at,
                   to_timestamp(
                       floor(extract(epoch FROM r.created_at) / {_SCHEDULE_PERIOD_SECONDS})
                       * {_SCHEDULE_PERIOD_SECONDS}
                   )
               ) AS bucket_at,
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
        GROUP BY r.provider, r.model, r.benchmark, r.metric_type,
                 COALESCE(
                     rn.scheduled_at,
                     to_timestamp(
                         floor(extract(epoch FROM r.created_at) / {_SCHEDULE_PERIOD_SECONDS})
                         * {_SCHEDULE_PERIOD_SECONDS}
                     )
                 )
    ) agg
"""  # noqa: S608


def upgrade() -> None:
    """Create results_by_bucket and backfill it from all of results."""
    op.execute(
        """
        CREATE TABLE benchmarks_v2.results_by_bucket (
            provider      TEXT NOT NULL,
            model         TEXT NOT NULL,
            benchmark     TEXT NOT NULL CHECK (benchmark IN ('STT','TTS')),
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
        """
    )
    # Serves the series read: benchmark equality + bucket_at range.
    op.execute(
        "CREATE INDEX results_by_bucket_series_idx "
        "ON benchmarks_v2.results_by_bucket (benchmark, bucket_at)"
    )
    op.execute(_BACKFILL_SQL)


def downgrade() -> None:
    """Drop the rollup table."""
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.results_by_bucket")
