# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Model and tag registry tables: models, tags, model_tags, model_history.

Revision ID: 20260824_0019
Revises:     20260818_0018
Create Date: 2026-08-24

The model registry (BENCH-710 design doc). No seed: rows are loaded through the
admin API, which validates them.

- ``models``         one row per benchmarked model. ``collected`` = the orchestrator
                     schedules runs for it; ``published`` = the public site shows it.
                     The surrogate ``id`` is the API resource id, so a rename does not
                     orphan history; the natural key stays unique.
- ``tags``           the MODE and FEATURES tag vocabulary. The other facet categories
                     (type, host, creator, source, licensing, deployment, region) are
                     derived from model columns at the API boundary.
- ``model_tags``     which vocabulary tags each model carries.
- ``model_history``  full before/after row snapshots per change. No FK to ``models``:
                     history must outlive anything it references.

Enum vocabularies (source, licensing, voice genders) are validated by pydantic at
the API boundary; only modality, region and tag category are constrained here.

Notes
-----
DB roles (``runner``, ``api``) and GRANTs are managed by Terraform. The ``api``
role needs INSERT and UPDATE on these four tables: admin writes happen in the
API process.
"""

from __future__ import annotations

from alembic import op

revision = "20260824_0019"
down_revision = "20260818_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the model and tag registry tables."""
    op.execute(
        """
        CREATE TABLE benchmarks_v2.models (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            modality      TEXT NOT NULL CHECK (modality IN ('STT', 'TTS', 'S2S')),
            provider      TEXT NOT NULL CHECK (provider <> ''),
            model         TEXT NOT NULL CHECK (model <> ''),
            voice         TEXT CHECK (voice IS NULL OR voice <> ''),          -- TTS synthesis voice
            voices        JSONB NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(voices) = 'array'), -- gendered pool [{id, gender, name, accent}]
            creator       TEXT CHECK (creator IS NULL OR creator <> ''),      -- NULL = same as provider
            source        TEXT NOT NULL DEFAULT 'official-api' CHECK (source <> ''),
            licensing     TEXT NOT NULL DEFAULT 'proprietary' CHECK (licensing <> ''),
            on_prem       BOOLEAN NOT NULL DEFAULT false,
            region        TEXT CHECK (region IN ('us', 'eu', 'asia')),        -- NULL = unknown
            arena_enabled BOOLEAN NOT NULL DEFAULT true,
            collected     BOOLEAN NOT NULL,
            published     BOOLEAN NOT NULL,
            updated_by_user_id TEXT NOT NULL CHECK (updated_by_user_id <> ''), -- Clerk sub
            updated_by_email   TEXT CHECK (updated_by_email IS NULL OR updated_by_email <> ''), -- label at write time; scrubbable
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (modality, provider, model)
        );

        CREATE TABLE benchmarks_v2.tags (
            value    TEXT PRIMARY KEY CHECK (value <> ''),
            category TEXT NOT NULL CHECK (category IN ('mode', 'features')),
            label    TEXT NOT NULL CHECK (label <> '')
        );

        CREATE TABLE benchmarks_v2.model_tags (
            model_id BIGINT NOT NULL REFERENCES benchmarks_v2.models(id) ON DELETE CASCADE,
            tag      TEXT NOT NULL REFERENCES benchmarks_v2.tags(value),
            PRIMARY KEY (model_id, tag)
        );

        CREATE TABLE benchmarks_v2.model_history (
            id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            model_id  BIGINT NOT NULL,
            modality  TEXT NOT NULL CHECK (modality IN ('STT', 'TTS', 'S2S')),
            provider  TEXT NOT NULL CHECK (provider <> ''),
            model     TEXT NOT NULL CHECK (model <> ''),
            old       JSONB CHECK (old IS NULL OR jsonb_typeof(old) = 'object'), -- NULL on create
            new       JSONB NOT NULL CHECK (jsonb_typeof(new) = 'object'),
            changed_by_user_id TEXT NOT NULL CHECK (changed_by_user_id <> ''),
            changed_by_org_id  TEXT CHECK (changed_by_org_id IS NULL OR changed_by_org_id <> ''), -- NULL = coval staff
            changed_by_email   TEXT CHECK (changed_by_email IS NULL OR changed_by_email <> ''),
            changed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX model_history_model_id_changed_at
            ON benchmarks_v2.model_history (model_id, changed_at DESC);
        """
    )


def downgrade() -> None:
    """Drop the registry tables."""
    op.execute(
        """
        DROP TABLE IF EXISTS benchmarks_v2.model_tags;
        DROP TABLE IF EXISTS benchmarks_v2.model_history;
        DROP TABLE IF EXISTS benchmarks_v2.models;
        DROP TABLE IF EXISTS benchmarks_v2.tags;
        """
    )
