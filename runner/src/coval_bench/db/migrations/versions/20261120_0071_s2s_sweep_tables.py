# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""S2S sweep tables.

Revision ID: 20261120_0071
Revises:     20261120_0070
Create Date: 2026-11-20

An S2S sweep is a ``sweeps`` row whose dataset is a scenario. Its items are
calls: one test case, one persona and a repeat number.

``s2s_queue`` is one row per call a sweep still has to make. ``claimed_at`` is
set while a worker has it, and ``simulation_id`` once Coval has the call and its
scores are still to come. A row is deleted when its call gets a result, and
stays at zero tries when the call failed every time.

``s2s_results`` is one row per finished call. ``s2s_scores`` is one row per
result, metric and metric version, as a ``value`` over ``reference_units`` so a
rate can be pooled across a whole sweep.
"""

from __future__ import annotations

from alembic import op

revision = "20261120_0071"
down_revision = "20261120_0070"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE benchmarks_v2.s2s_queue (
            sweep_id      BIGINT NOT NULL REFERENCES benchmarks_v2.sweeps (id),
            test_case     TEXT NOT NULL CHECK (test_case <> ''),
            persona       TEXT NOT NULL CHECK (persona <> ''),
            repeat        INTEGER NOT NULL CHECK (repeat >= 1),
            tries_left    INTEGER NOT NULL DEFAULT 3 CHECK (tries_left >= 0),
            last_error    TEXT CHECK (last_error <> ''),
            claimed_at    TIMESTAMPTZ,
            simulation_id TEXT UNIQUE CHECK (simulation_id <> ''),
            PRIMARY KEY (sweep_id, test_case, persona, repeat),
            CHECK (simulation_id IS NULL OR claimed_at IS NOT NULL)
        );

        CREATE TABLE benchmarks_v2.s2s_results (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            sweep_id      BIGINT NOT NULL REFERENCES benchmarks_v2.sweeps (id),
            test_case     TEXT NOT NULL CHECK (test_case <> ''),
            persona       TEXT NOT NULL CHECK (persona <> ''),
            repeat        INTEGER NOT NULL CHECK (repeat >= 1),
            simulation_id TEXT NOT NULL UNIQUE CHECK (simulation_id <> ''),
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (sweep_id, test_case, persona, repeat)
        );

        CREATE TABLE benchmarks_v2.s2s_scores (
            id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            s2s_result_id   BIGINT NOT NULL REFERENCES benchmarks_v2.s2s_results (id),
            metric          TEXT NOT NULL CHECK (metric <> ''),
            metric_version  TEXT NOT NULL CHECK (metric_version <> ''),
            value           DOUBLE PRECISION NOT NULL,
            reference_units INTEGER NOT NULL CHECK (reference_units >= 0),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (s2s_result_id, metric, metric_version)
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS benchmarks_v2.s2s_scores;
        DROP TABLE IF EXISTS benchmarks_v2.s2s_results;
        DROP TABLE IF EXISTS benchmarks_v2.s2s_queue;
        """
    )
