# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the org that changed a model or a company.

Revision ID: 20261119_0069
Revises:     20261117_0068
Create Date: 2026-11-19

A change made with an org's API key has an org but no user, so the user becomes
optional and the org gets its own column. Every row still names one of the two.
``model_history`` already has the org column.
"""

from __future__ import annotations

from alembic import op

revision = "20261119_0069"
down_revision = "20261117_0068"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the org columns and let a row name an org without a user."""
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models
            ADD COLUMN updated_by_org_id TEXT CHECK (updated_by_org_id <> ''),
            ALTER COLUMN updated_by_user_id DROP NOT NULL,
            ADD CONSTRAINT models_updated_by_someone
                CHECK (updated_by_user_id IS NOT NULL OR updated_by_org_id IS NOT NULL);
        ALTER TABLE benchmarks_v2.companies
            ADD COLUMN updated_by_org_id TEXT CHECK (updated_by_org_id <> ''),
            ALTER COLUMN updated_by_user_id DROP NOT NULL,
            ADD CONSTRAINT companies_updated_by_someone
                CHECK (updated_by_user_id IS NOT NULL OR updated_by_org_id IS NOT NULL);
        ALTER TABLE benchmarks_v2.model_history
            ALTER COLUMN changed_by_user_id DROP NOT NULL,
            ADD CONSTRAINT model_history_changed_by_someone
                CHECK (changed_by_user_id IS NOT NULL OR changed_by_org_id IS NOT NULL);
        """
    )


def downgrade() -> None:
    """Put the org in the user column where there is no user, then drop the org columns."""
    op.execute(
        """
        ALTER TABLE benchmarks_v2.model_history
            DROP CONSTRAINT model_history_changed_by_someone;
        UPDATE benchmarks_v2.model_history SET changed_by_user_id = changed_by_org_id
            WHERE changed_by_user_id IS NULL;
        ALTER TABLE benchmarks_v2.model_history ALTER COLUMN changed_by_user_id SET NOT NULL;

        ALTER TABLE benchmarks_v2.companies DROP CONSTRAINT companies_updated_by_someone;
        UPDATE benchmarks_v2.companies SET updated_by_user_id = updated_by_org_id
            WHERE updated_by_user_id IS NULL;
        ALTER TABLE benchmarks_v2.companies
            ALTER COLUMN updated_by_user_id SET NOT NULL,
            DROP COLUMN updated_by_org_id;

        ALTER TABLE benchmarks_v2.models DROP CONSTRAINT models_updated_by_someone;
        UPDATE benchmarks_v2.models SET updated_by_user_id = updated_by_org_id
            WHERE updated_by_user_id IS NULL;
        ALTER TABLE benchmarks_v2.models
            ALTER COLUMN updated_by_user_id SET NOT NULL,
            DROP COLUMN updated_by_org_id;
        """
    )
