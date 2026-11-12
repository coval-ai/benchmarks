# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Writes for ``benchmarks_v2.sweeps``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from psycopg_pool import AsyncConnectionPool

_ENSURE = """
    INSERT INTO benchmarks_v2.sweeps (model_id, model_version, dataset_id, dataset_version)
    SELECT m.id, m.version, %(dataset_id)s, %(dataset_version)s
    FROM benchmarks_v2.models m
    WHERE m.id = %(model_id)s
    ON CONFLICT (model_id, model_version, dataset_id, dataset_version) DO NOTHING
"""


async def ensure_sweeps(
    pool: AsyncConnectionPool[Any], wanted: Sequence[tuple[int, str, str]]
) -> int:
    """Open a sweep, at the model's current version, for each (model id, dataset id,
    dataset version) that has none. Returns how many were opened."""
    opened = 0
    async with pool.connection() as conn, conn.cursor() as cur:
        for model_id, dataset_id, dataset_version in wanted:
            await cur.execute(
                _ENSURE,
                {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "dataset_version": dataset_version,
                },
            )
            opened += cur.rowcount
    return opened
