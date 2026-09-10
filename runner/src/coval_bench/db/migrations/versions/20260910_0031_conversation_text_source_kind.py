# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Admit ``conversation_text`` as an observation source kind.

LLM conversations have no audio, so the normalized store needs a text source
kind before the LLM fetch can dual-write. The benchmark CHECKs already admit
'LLM' since 20260901_0025; this widens the one remaining constraint.
"""

from __future__ import annotations

import time

import psycopg.errors
import sqlalchemy.exc
from alembic import op

revision = "20260910_0031"
down_revision = "20260910_0030"
branch_labels = None
depends_on = None

_CONSTRAINT = "benchmark_observations_source_kind_check"
_WIDE = "'dataset_audio','generated_audio','conversation_audio','conversation_text'"
_NARROW = "'dataset_audio','generated_audio','conversation_audio'"

_LOCK_TIMEOUT = "10s"
_LOCK_ATTEMPTS = 12
_RETRY_PAUSE_SECONDS = 5.0


def _execute_retrying_locks(sql: str) -> None:
    for attempt in range(1, _LOCK_ATTEMPTS + 1):
        try:
            op.execute(sql)
            return
        except sqlalchemy.exc.OperationalError as exc:
            lock_timeout = isinstance(exc.orig, psycopg.errors.LockNotAvailable)
            if not lock_timeout or attempt == _LOCK_ATTEMPTS:
                raise
            time.sleep(_RETRY_PAUSE_SECONDS)


def _swap_constraint(values: str) -> None:
    with op.get_context().autocommit_block():
        op.execute(f"SET lock_timeout = '{_LOCK_TIMEOUT}'")
        _execute_retrying_locks(
            f"ALTER TABLE benchmarks_v2.benchmark_observations"
            f" DROP CONSTRAINT IF EXISTS {_CONSTRAINT}"
        )
        _execute_retrying_locks(
            f"ALTER TABLE benchmarks_v2.benchmark_observations"
            f" ADD CONSTRAINT {_CONSTRAINT} CHECK (source_kind IN ({values})) NOT VALID"
        )
        _execute_retrying_locks(
            f"ALTER TABLE benchmarks_v2.benchmark_observations VALIDATE CONSTRAINT {_CONSTRAINT}"
        )
        op.execute("RESET lock_timeout")


def upgrade() -> None:
    _swap_constraint(_WIDE)


def downgrade() -> None:
    """Re-narrow; fails while conversation_text rows remain."""
    _swap_constraint(_NARROW)
