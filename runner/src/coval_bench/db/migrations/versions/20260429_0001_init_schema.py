# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Initial schema: benchmarks_v2.runs, benchmarks_v2.results, results_24h MV.

Revision ID: 20260429_0001
Revises:     (none — initial migration)
Create Date: 2026-04-29

Notes
-----
DB roles (``runner``, ``api``) and GRANTs are managed by Terraform.
Alembic does NOT create or alter roles.

The ``results_24h`` materialized view is populated immediately but will be
empty until results are inserted.  Refresh is handled out-of-band by a
Cloud Scheduler cron via ``REFRESH MATERIALIZED VIEW CONCURRENTLY
results_24h``. That cron is out of scope for this migration.
"""

from __future__ import annotations

from alembic import op

revision = "20260429_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create schema, tables, indexes, and materialized view."""
    op.execute("CREATE SCHEMA IF NOT EXISTS benchmarks_v2")
    op.execute("SET search_path TO benchmarks_v2")

    op.execute(
        """
        CREATE TABLE runs (
            id              BIGSERIAL PRIMARY KEY,
            started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at     TIMESTAMPTZ,
            runner_sha      TEXT NOT NULL,
            dataset_id      TEXT NOT NULL,
            dataset_sha256  TEXT NOT NULL,
            status          TEXT NOT NULL
                            CHECK (status IN ('running','succeeded','partial','failed')),
            error           TEXT
        )
        """
    )
    op.execute("CREATE INDEX runs_started_at_idx ON runs (started_at DESC)")

    op.execute(
        """
        CREATE TABLE results (
            id              BIGSERIAL PRIMARY KEY,
            run_id          BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            provider        TEXT NOT NULL,
            model           TEXT NOT NULL,
            voice           TEXT,
            benchmark       TEXT NOT NULL CHECK (benchmark IN ('STT','TTS')),
            metric_type     TEXT NOT NULL,
            metric_value    DOUBLE PRECISION,
            metric_units    TEXT,
            audio_filename  TEXT,
            transcript      TEXT,
            status          TEXT NOT NULL CHECK (status IN ('success','failed')),
            error           TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX results_run_id_idx ON results (run_id)")
    op.execute(
        "CREATE INDEX results_provider_model_idx "
        "ON results (provider, model, metric_type, created_at DESC)"
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW results_24h AS
        SELECT provider, model, benchmark, metric_type,
               avg(metric_value)                                        AS avg_value,
               percentile_cont(0.5)  WITHIN GROUP (ORDER BY metric_value) AS p50,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY metric_value) AS p95,
               count(*)                                                 AS n
        FROM results
        WHERE status = 'success'
          AND created_at >= now() - INTERVAL '24 hours'
        GROUP BY provider, model, benchmark, metric_type
        """
    )
    op.execute(
        "CREATE INDEX results_24h_lookup_idx ON results_24h (benchmark, metric_type, avg_value)"
    )


def downgrade() -> None:
    """Drop materialized view, tables, and schema (in dependency order)."""
    op.execute("SET search_path TO benchmarks_v2")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS results_24h")
    op.execute("DROP TABLE IF EXISTS results")
    op.execute("DROP TABLE IF EXISTS runs")
    op.execute("DROP SCHEMA IF EXISTS benchmarks_v2 CASCADE")
