# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Delete Murf rows recorded before falcon-2 was actually served.

Murf's us-east cluster silently substitutes Gen2 when it does not recognise
the requested model value, so every ``murf/falcon-2`` row scheduled before
2026-08-07 20:00 UTC measured Gen2 under falcon-2's key. They misstate the
model's latency several-fold and would pollute its aggregates, so they are
removed rather than annotated.

``results`` carries no timestamp of its own; the cutoff is applied through
the owning run's ``scheduled_at``. ``results_by_bucket`` is keyed by
``bucket_at`` directly. The per-window matviews are deliberately not
refreshed here: the runner refreshes them at the end of each benchmark run
(see ``migrations/env.py``), and ``REFRESH ... CONCURRENTLY`` cannot run
inside a migration's transaction.
"""

from __future__ import annotations

from alembic import op

revision = "20260807_0015"
down_revision = "20260804_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Remove the mislabeled Gen2 rows from both results tables."""
    op.execute(
        "DELETE FROM benchmarks_v2.results r "
        "USING benchmarks_v2.runs ru "
        "WHERE r.run_id = ru.id "
        "AND r.provider = 'murf' AND r.model = 'falcon-2' "
        "AND ru.scheduled_at < '2026-08-07 20:00:00+00'"
    )
    op.execute(
        "DELETE FROM benchmarks_v2.results_by_bucket "
        "WHERE provider = 'murf' AND model = 'falcon-2' "
        "AND bucket_at < '2026-08-07 20:00:00+00'"
    )


def downgrade() -> None:
    """The deleted rows are unrecoverable; nothing to restore."""
