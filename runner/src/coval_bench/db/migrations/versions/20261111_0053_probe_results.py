# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Latency probe results.

Revision ID: 20261111_0053
Revises:     20261110_0052
Create Date: 2026-11-11

One row per model per tick from the latency probe, which sends every model the
same short input. Timings are milliseconds. A column is NULL when the metric
doesn't apply to the modality or the probe failed.
"""

from __future__ import annotations

from alembic import op

revision = "20261111_0053"
down_revision = "20261110_0052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE benchmarks_v2.probe_results (
            id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            benchmark    TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS')),
            provider     TEXT NOT NULL CHECK (provider <> ''),
            model        TEXT NOT NULL CHECK (model <> ''),
            voice        TEXT CHECK (voice IS NULL OR voice <> ''),
            scheduled_at TIMESTAMPTZ NOT NULL,
            sent_at      TIMESTAMPTZ NOT NULL,
            ttft_ms      DOUBLE PRECISION CHECK (ttft_ms >= 0 AND ttft_ms <> 'Infinity'::float8),
            ttfs_ms      DOUBLE PRECISION CHECK (ttfs_ms <> 'NaN'::float8 AND abs(ttfs_ms) <> 'Infinity'::float8),
            ttfa_ms      DOUBLE PRECISION CHECK (ttfa_ms >= 0 AND ttfa_ms <> 'Infinity'::float8),
            status       TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
            error        TEXT,
            failure_origin TEXT CHECK (failure_origin IN ('provider', 'runner')),
            CHECK ((status = 'succeeded' AND error IS NULL AND failure_origin IS NULL) OR (status = 'failed' AND error IS NOT NULL AND error <> '' AND failure_origin IS NOT NULL)),
            UNIQUE (benchmark, provider, model, scheduled_at)
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.probe_results;")
