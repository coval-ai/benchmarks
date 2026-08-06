# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture raw usage quantities in native billing units.

Per-result quantities land on ``results`` (tokens, provider-billed seconds,
input characters, measured audio durations) and whisper-1 judge totals land on
``runs``. All nullable, no backfill: providers report what they report, and
cost math over these columns is a later migration's concern. The stats
matviews don't select any of these columns, so unlike 0014 they are left
untouched — a plain ADD COLUMN takes only the one lock and cannot deadlock
against a concurrent matview refresh.
"""

from __future__ import annotations

from alembic import op

revision = "20260806_0015"
down_revision = "20260804_0014"
branch_labels = None
depends_on = None

_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "results": (
        ("input_tokens", "BIGINT"),
        ("output_tokens", "BIGINT"),
        ("total_tokens", "BIGINT"),
        ("billable_seconds", "DOUBLE PRECISION"),
        ("characters_in", "INTEGER"),
        ("audio_seconds_in", "DOUBLE PRECISION"),
        ("audio_seconds_out", "DOUBLE PRECISION"),
    ),
    "runs": (
        # whisper-1 bills by audio duration and reports duration-type usage, so
        # the judge's real billing unit is seconds; the token columns cover a
        # future token-billed judge model.
        ("judge_input_tokens", "BIGINT"),
        ("judge_output_tokens", "BIGINT"),
        ("judge_audio_seconds", "DOUBLE PRECISION"),
    ),
}


def upgrade() -> None:
    for table, columns in _COLUMNS.items():
        for name, sql_type in columns:
            op.execute(f"ALTER TABLE benchmarks_v2.{table} ADD COLUMN {name} {sql_type}")


def downgrade() -> None:
    for table, columns in _COLUMNS.items():
        for name, _ in columns:
            op.execute(f"ALTER TABLE benchmarks_v2.{table} DROP COLUMN {name}")
