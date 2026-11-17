# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Benchmark registry tables: benchmarks and benchmark_datasets.

Revision ID: 20261117_0065
Revises:     20261116_0064
Create Date: 2026-11-17

Says which benchmark and condition each dataset id belongs to. No seed: rows are
entered through the admin API, the way models are.

- ``benchmarks``          one row per named test, with its modality.
- ``benchmark_datasets``  the dataset ids a benchmark owns. A dataset id belongs to one
                          benchmark under one condition; NULL is a benchmark with no conditions.

Result rows are untouched and keep their ``dataset_id``.
"""

from __future__ import annotations

from alembic import op

revision = "20261117_0065"
down_revision = "20261116_0064"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the benchmark registry tables."""
    op.execute(
        """
        CREATE TABLE benchmarks_v2.benchmarks (
            slug      TEXT PRIMARY KEY CHECK (slug ~ '^[a-z0-9]+(-[a-z0-9]+)*$'),
            modality  TEXT NOT NULL CHECK (modality IN ('STT', 'TTS', 'S2S', 'LLM')),
            name      TEXT NOT NULL CHECK (name <> ''),
            updated_by_user_id TEXT NOT NULL CHECK (updated_by_user_id <> ''),
            updated_by_email   TEXT CHECK (updated_by_email IS NULL OR updated_by_email <> ''),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE benchmarks_v2.benchmark_datasets (
            dataset_id TEXT PRIMARY KEY CHECK (dataset_id <> ''),
            benchmark  TEXT NOT NULL REFERENCES benchmarks_v2.benchmarks(slug),
            condition  TEXT CHECK (condition IS NULL OR condition ~ '^[a-z0-9]+(-[a-z0-9]+)*$'),
            UNIQUE NULLS NOT DISTINCT (benchmark, condition)
        );
        """
    )


def downgrade() -> None:
    """Drop the benchmark registry tables."""
    op.execute(
        """
        DROP TABLE IF EXISTS benchmarks_v2.benchmark_datasets;
        DROP TABLE IF EXISTS benchmarks_v2.benchmarks;
        """
    )
