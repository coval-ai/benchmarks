# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""S2S agent kind: models.architecture and models.orchestration.

Revision ID: 20261028_0046
Revises:     20261010_0045
Create Date: 2026-10-28

The S2S board is gaining agents that are not one speech-to-speech model: cascaded
agents run by a voice agent platform, and models run inside the Coval harness.
``architecture`` says how the agent makes speech; ``orchestration`` says who runs
its turn-taking and tools. Both are S2S-only, so every other modality keeps NULL.

Every S2S row today is a speech-to-speech model Coval calls directly, so the seed
sets ``architecture`` and leaves ``orchestration`` NULL.
"""

from __future__ import annotations

from alembic import op

revision = "20261028_0046"
down_revision = "20261010_0045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models
            ADD COLUMN architecture TEXT
                CHECK (architecture IN ('speech-to-speech', 'cascaded')),
            ADD COLUMN orchestration TEXT
                CHECK (orchestration IN ('harness', 'platform')),
            ADD CONSTRAINT models_agent_kind_s2s_only CHECK (
                modality = 'S2S' OR (architecture IS NULL AND orchestration IS NULL)
            );
        """
    )
    op.execute(
        "UPDATE benchmarks_v2.models SET architecture = 'speech-to-speech' WHERE modality = 'S2S';"
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models
            DROP CONSTRAINT models_agent_kind_s2s_only,
            DROP COLUMN orchestration,
            DROP COLUMN architecture;
        """
    )
