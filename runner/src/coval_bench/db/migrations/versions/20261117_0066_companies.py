# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Companies, each with the Clerk org that belongs to it.

Revision ID: 20261117_0066
Revises:     20261117_0065
Create Date: 2026-11-17

One row per provider or creator. The slug is the string models already carry, so
nothing stored is renamed. Filled from the models table; later rows come through the
admin API.

``clerk_org_id`` is the company's Clerk org, which sees the company's unpublished
models. A company has at most one, and an org belongs to at most one company.

Models get no foreign key to ``companies`` yet. Nothing reads the table yet.
"""

from __future__ import annotations

from alembic import op

revision = "20261117_0066"
down_revision = "20261117_0065"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the table and a company for every provider and creator on a model."""
    op.execute(
        """
        CREATE TABLE benchmarks_v2.companies (
            slug TEXT PRIMARY KEY CHECK (slug <> ''),
            name TEXT NOT NULL CHECK (name <> ''),
            clerk_org_id TEXT UNIQUE CHECK (clerk_org_id <> ''),
            updated_by_user_id TEXT NOT NULL CHECK (updated_by_user_id <> ''),
            updated_by_email   TEXT CHECK (updated_by_email IS NULL OR updated_by_email <> ''),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        INSERT INTO benchmarks_v2.companies (slug, name, updated_by_user_id)
        SELECT slug, slug, 'migration'
        FROM (
            SELECT provider AS slug FROM benchmarks_v2.models
            UNION
            SELECT creator FROM benchmarks_v2.models WHERE creator IS NOT NULL
        ) known;
        """
    )


def downgrade() -> None:
    """Drop the table."""
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.companies")
