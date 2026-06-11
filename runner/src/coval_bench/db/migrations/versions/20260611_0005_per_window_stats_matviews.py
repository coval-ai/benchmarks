# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-window stats materialized views: results_24h, results_7d, results_30d.

One view per dashboard window, all sharing a wide stats shape per
(provider, model, benchmark, metric_type): avg, sample stddev, min,
p25/p50/p75/p90/p95/p99 (a single PERCENTILE_CONT over an array, so one
sort), max, count. Rows gate on result status='success', a non-null metric_value,
and a parent run in (succeeded, partial) — the same filters the aggregates
endpoint applied live.

results_24h is dropped and recreated in this shape (it previously held only
avg/p50/p95/n and did not gate on the parent run). Each view gets a unique
index on the group key so it can be refreshed with REFRESH MATERIALIZED VIEW
CONCURRENTLY. The runner refreshes them at the end of each benchmark run; this
migration only creates the views, so they hold data as of migration time until
the first post-migration run refreshes them.
"""

from __future__ import annotations

from alembic import op

revision = "20260611_0005"
down_revision = "20260605_0004"
branch_labels = None
depends_on = None

_WINDOWS: dict[str, str] = {
    "results_24h": "24 hours",
    "results_7d": "7 days",
    "results_30d": "30 days",
}


def _view_sql(name: str, interval: str) -> str:
    # S608 false-positive: name and interval come from the _WINDOWS constant.
    return f"""
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
    """  # noqa: S608


def upgrade() -> None:
    """Recreate results_24h in the wide shape and add results_7d/results_30d."""
    op.execute("DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.results_24h")
    for name, interval in _WINDOWS.items():
        op.execute(_view_sql(name, interval))
        op.execute(
            f"CREATE UNIQUE INDEX {name}_group_key "
            f"ON benchmarks_v2.{name} (provider, model, benchmark, metric_type)"
        )


def downgrade() -> None:
    """Drop the per-window views and restore the original results_24h."""
    for name in _WINDOWS:
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.{name}")
    op.execute(
        """
        CREATE MATERIALIZED VIEW benchmarks_v2.results_24h AS
        SELECT provider, model, benchmark, metric_type,
               avg(metric_value)                                        AS avg_value,
               percentile_cont(0.5)  WITHIN GROUP (ORDER BY metric_value) AS p50,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY metric_value) AS p95,
               count(*)                                                 AS n
        FROM benchmarks_v2.results
        WHERE status = 'success'
          AND created_at >= now() - INTERVAL '24 hours'
        GROUP BY provider, model, benchmark, metric_type
        """
    )
    op.execute(
        "CREATE INDEX results_24h_lookup_idx "
        "ON benchmarks_v2.results_24h (benchmark, metric_type, avg_value)"
    )
