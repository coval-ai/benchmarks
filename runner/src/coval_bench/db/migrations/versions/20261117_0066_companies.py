# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Companies and the Clerk orgs that belong to them.

Revision ID: 20261117_0066
Revises:     20261117_0065
Create Date: 2026-11-17

- ``companies``     one row per provider or creator. The slug is the string models
                    already carry, so nothing stored is renamed. Filled from the models
                    table; later rows come through the admin API.
- ``company_orgs``  links a Clerk org to a company. The org sees that company's
                    unpublished models.

Models get no foreign key to ``companies`` yet. Nothing reads these tables yet.
"""

from __future__ import annotations

from alembic import op

revision = "20261117_0066"
down_revision = "20261117_0065"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the tables and a company for every provider and creator on a model."""
    op.execute(
        """
        CREATE TABLE benchmarks_v2.companies (
            slug TEXT PRIMARY KEY CHECK (slug <> ''),
            name TEXT NOT NULL CHECK (name <> ''),
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

        CREATE TABLE benchmarks_v2.company_orgs (
            org_id    TEXT PRIMARY KEY CHECK (org_id <> ''),
            company   TEXT NOT NULL REFERENCES benchmarks_v2.companies(slug),
            added_by_user_id TEXT NOT NULL CHECK (added_by_user_id <> ''),
            added_by_email   TEXT CHECK (added_by_email IS NULL OR added_by_email <> ''),
            added_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX company_orgs_company ON benchmarks_v2.company_orgs (company);
        """
    )


def downgrade() -> None:
    """Drop the tables."""
    op.execute(
        """
        DROP TABLE IF EXISTS benchmarks_v2.company_orgs;
        DROP TABLE IF EXISTS benchmarks_v2.companies;
        """
    )
