# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice Arena lean schema: arena.battles, arena.votes, arena.leaderboard_snapshots.

Revision ID: 20260615_0007
Revises:     20260611_0006
Create Date: 2026-06-15

Three tables in the ``arena`` schema, alongside ``benchmarks_v2`` in the same DB:

- ``battles``  raw matchups. Model identity is stored as text (provider + model) using the
               same keys as ``benchmarks_v2.results``, so arena rows can join to benchmark
               metrics without a shared id. Domain and prompt text are inlined.
- ``votes``    raw human judgments (A_WIN/B_WIN/TIE). ``voter_type`` separates internal
               ``labeler`` votes from ``external`` (public) votes; ``UNIQUE (battle_id,
               voter_type, voter_id)`` keeps one current vote per identity (re-label = UPDATE
               that row). Raw votes are the source of truth — Bradley-Terry / Davidson are refit
               from these at any time, for any window; nothing here is ever an input from a
               previously computed rating.
- ``leaderboard_snapshots``  persisted computed ratings: one row per model per computation run
               (a board = all rows sharing ``computed_at`` + ``metric_name`` +
               ``methodology_version`` + ``domain``). This is a cache/history of derived data,
               refreshed from votes — never an input to the rating math.

Notes
-----
DB roles (``runner``, ``api``) and GRANTs are managed by Terraform. Alembic does NOT create or
alter roles; the roles must be granted USAGE on ``arena`` and privileges on its tables in
Terraform before the API can read or write here.

Primary keys are UUID defaulted with ``gen_random_uuid()`` (core since Postgres 13).
"""

from __future__ import annotations

from alembic import op

revision = "20260615_0007"
down_revision = "20260611_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the arena schema and its three tables."""
    op.execute("CREATE SCHEMA IF NOT EXISTS arena")
    op.execute("SET search_path TO arena")

    op.execute(
        """
        CREATE TABLE battles (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            provider_a  TEXT NOT NULL,
            model_a     TEXT NOT NULL,
            provider_b  TEXT NOT NULL,
            model_b     TEXT NOT NULL,
            domain      TEXT,
            prompt_text TEXT NOT NULL,
            audio_a_url TEXT NOT NULL,
            audio_b_url TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX battles_domain_idx ON battles (domain)")

    op.execute(
        """
        CREATE TABLE votes (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            battle_id  UUID NOT NULL REFERENCES battles(id),
            outcome    TEXT NOT NULL CHECK (outcome IN ('A_WIN','B_WIN','TIE')),
            voter_type TEXT NOT NULL CHECK (voter_type IN ('labeler','external')),
            voter_id   TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (battle_id, voter_type, voter_id)
        )
        """
    )
    op.execute("CREATE INDEX votes_battle_id_idx ON votes (battle_id)")

    op.execute(
        """
        CREATE TABLE leaderboard_snapshots (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            computed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            metric_name         TEXT NOT NULL,
            methodology_version TEXT NOT NULL,
            domain              TEXT,
            provider            TEXT NOT NULL,
            model               TEXT NOT NULL,
            rating_elo          NUMERIC NOT NULL,
            rating_bt           NUMERIC NOT NULL,
            ci_low              NUMERIC,
            ci_high             NUMERIC,
            ci_half_width       NUMERIC,
            votes_total         INTEGER NOT NULL,
            wins                NUMERIC NOT NULL,
            losses              NUMERIC NOT NULL,
            ties                INTEGER NOT NULL,
            status              TEXT NOT NULL
                                CHECK (status IN ('preliminary','usable','established'))
        )
        """
    )
    op.execute(
        "CREATE INDEX leaderboard_snapshots_lookup_idx "
        "ON leaderboard_snapshots (metric_name, computed_at)"
    )


def downgrade() -> None:
    """Drop the entire arena schema and everything in it."""
    op.execute("DROP SCHEMA IF EXISTS arena CASCADE")
