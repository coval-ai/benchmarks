# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Add per-turn timing captured by the Phonely LLM proxy."""

from __future__ import annotations

from alembic import op

revision = "20260902_0026"
down_revision = "20260901_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE benchmarks_v2.llm_turns (
            id            BIGSERIAL PRIMARY KEY,
            simulation_id TEXT NOT NULL CHECK (simulation_id <> ''),
            turn_index    INTEGER NOT NULL CHECK (turn_index >= 0),
            provider      TEXT NOT NULL CHECK (provider <> ''),
            model         TEXT NOT NULL CHECK (model <> ''),
            ttft_ms       DOUBLE PRECISION NOT NULL CHECK (
                ttft_ms >= 0 AND ttft_ms NOT IN (
                    'NaN'::float8, 'Infinity'::float8, '-Infinity'::float8
                )
            ),
            total_ms      DOUBLE PRECISION NOT NULL CHECK (total_ms >= ttft_ms),
            output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX llm_turns_simulation_id ON benchmarks_v2.llm_turns (simulation_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.llm_turns")
