# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add http_version and submit_to_headers_ms diagnostics to results.

Revision ID: 20260601_0002
Revises:     20260429_0001
Create Date: 2026-06-01

These columns let analysis filter HTTP TTS rows whose TTFA reabsorbed
connection setup: ``http_version`` distinguishes HTTP/2 from HTTP/1.1
fallback, and ``submit_to_headers_ms`` flags rows where the warm pool
reconnected mid-run.

Deferred / out of scope: the ``results_24h`` materialized view is not changed
to filter on ``http_version``. Contaminated rows are already excluded from
aggregates because the orchestrator writes them with ``status='failed'`` and
the view filters ``status='success'``. Adding an explicit per-protocol filter
to the view (and to dashboard queries) is tracked as a separate follow-up.
"""

from __future__ import annotations

from alembic import op

revision = "20260601_0002"
down_revision = "20260429_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the two diagnostic columns to ``results``."""
    op.execute("SET search_path TO benchmarks_v2")
    op.execute("ALTER TABLE results ADD COLUMN http_version TEXT")
    op.execute("ALTER TABLE results ADD COLUMN submit_to_headers_ms DOUBLE PRECISION")


def downgrade() -> None:
    """Drop the two diagnostic columns."""
    op.execute("SET search_path TO benchmarks_v2")
    op.execute("ALTER TABLE results DROP COLUMN IF EXISTS submit_to_headers_ms")
    op.execute("ALTER TABLE results DROP COLUMN IF EXISTS http_version")
