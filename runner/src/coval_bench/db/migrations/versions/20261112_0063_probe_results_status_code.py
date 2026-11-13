# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe results status code.

Revision ID: 20261112_0063
Revises:     20261112_0062
Create Date: 2026-11-12

The HTTP status a failed probe got back, when the provider reported one. NULL
for a success and for a failure with no status, like a timeout.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0063"
down_revision = "20261112_0062"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE benchmarks_v2.probe_results ADD COLUMN status_code INTEGER;")


def downgrade() -> None:
    op.execute("ALTER TABLE benchmarks_v2.probe_results DROP COLUMN status_code;")
