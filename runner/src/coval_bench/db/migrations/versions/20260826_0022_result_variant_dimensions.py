# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the result variant dimensions and the mock tool-call log.

Orchestration platforms are measured under the existing ``S2S`` benchmark rather
than one of their own: they carry the same turn-latency and interruption metrics,
read from the same stereo recording, so the platform is a dimension of a row and
not a different experiment. ``variant_id`` is that dimension.

``variant_id`` is NOT NULL and defaults to ``pinned`` — the arm whose components
Coval fixed for comparability. Nullable would leave "unset" ambiguous between a
row predating variants and one somebody forgot to tag, and only the second is a
bug. Every existing row is Coval-pinned, so the default is also the backfill.

``transport``, ``test_case_id`` and ``iteration`` stay nullable and cannot be
backfilled, which is why this lands before the first row worth keeping.

``mock_tool_calls`` is deliberately its own table rather than rows in ``results``:
a tool call is an event carrying arguments and a response, not a numeric metric,
and ``results.metric_value`` is DOUBLE PRECISION with the rollups filtering on it
being non-null. Metrics *derived* from these rows still land in ``results``.
"""

from __future__ import annotations

from alembic import op

revision = "20260826_0022"
down_revision = "20260825_0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the result dimensions and the tool-call log."""
    # The default is also the backfill: every row written before variants existed
    # ran the components Coval pinned. A platform arm must set the column at write
    # time rather than lean on this.
    op.execute(
        """
        ALTER TABLE benchmarks_v2.results
            ADD COLUMN variant_id   TEXT NOT NULL DEFAULT 'pinned',
            ADD COLUMN transport    TEXT,
            ADD COLUMN test_case_id TEXT,
            ADD COLUMN iteration    INTEGER
        """
    )

    op.execute(
        """
        CREATE TABLE benchmarks_v2.mock_tool_calls (
            id             BIGSERIAL PRIMARY KEY,
            simulation_id  TEXT,
            caller_number  TEXT,
            tool           TEXT NOT NULL CHECK (tool <> ''),
            args           JSONB NOT NULL CHECK (jsonb_typeof(args) = 'object'),
            matched_seed   TEXT,
            response       JSONB NOT NULL,
            latency_ms     DOUBLE PRECISION NOT NULL
                           CHECK (latency_ms >= 0 AND latency_ms NOT IN (
                               'NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)),
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # The ingest join: every tool call for one simulation, in order.
    op.execute(
        "CREATE INDEX mock_tool_calls_simulation_idx "
        "ON benchmarks_v2.mock_tool_calls (simulation_id, created_at)"
    )
    # Correlation fallback for platforms that drop unknown SIP headers.
    op.execute(
        "CREATE INDEX mock_tool_calls_caller_idx "
        "ON benchmarks_v2.mock_tool_calls (caller_number, created_at) "
        "WHERE caller_number IS NOT NULL"
    )


def downgrade() -> None:
    """Drop the tool-call log and the result dimensions."""
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.mock_tool_calls")
    op.execute(
        """
        ALTER TABLE benchmarks_v2.results
            DROP COLUMN IF EXISTS iteration,
            DROP COLUMN IF EXISTS test_case_id,
            DROP COLUMN IF EXISTS transport,
            DROP COLUMN IF EXISTS variant_id
        """
    )
