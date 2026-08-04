# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rename the xAI S2S board key to name its model version.

``grok-realtime`` carried no version while its sibling entry names one
explicitly, so the two xAI rows read as unrelated products rather than as
1.0 and 2.0 of the same one.

``model`` is stored on every results row and is part of the
``results_by_bucket`` primary key, so both tables are rewritten here.
Rewriting only one would split the S2S timeline into two half-series, with
the leaderboard and the timeline disagreeing about which key is live.

The per-window matviews are deliberately not refreshed here: the runner
refreshes them at the end of each benchmark run (see ``migrations/env.py``),
and ``REFRESH ... CONCURRENTLY`` cannot run inside a migration's transaction.
"""

from __future__ import annotations

from alembic import op

revision = "20260803_0013"
down_revision = "20260727_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Point historical xAI S2S rows at the versioned key."""
    op.execute(
        "UPDATE benchmarks_v2.results "
        "SET model = 'grok-voice-think-fast-1.0' "
        "WHERE provider = 'xai' AND model = 'grok-realtime'"
    )
    op.execute(
        "UPDATE benchmarks_v2.results_by_bucket "
        "SET model = 'grok-voice-think-fast-1.0' "
        "WHERE provider = 'xai' AND model = 'grok-realtime'"
    )


def downgrade() -> None:
    """Restore the version-less key."""
    op.execute(
        "UPDATE benchmarks_v2.results "
        "SET model = 'grok-realtime' "
        "WHERE provider = 'xai' AND model = 'grok-voice-think-fast-1.0'"
    )
    op.execute(
        "UPDATE benchmarks_v2.results_by_bucket "
        "SET model = 'grok-realtime' "
        "WHERE provider = 'xai' AND model = 'grok-voice-think-fast-1.0'"
    )
