# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A lock on one model that holds across jobs, so two of them never call it at once."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    from coval_bench.registries import RegisteredModel

_ACQUIRE = "SELECT pg_advisory_lock(hashtextextended(%s, 0))"
_TRY_ACQUIRE = "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS held"
_RELEASE = "SELECT pg_advisory_unlock(hashtextextended(%s, 0))"


def model_lock_key(entry: RegisteredModel, provider_cls: Any) -> str:  # noqa: ANN401
    """The model, or its provider when the provider only takes one connection."""
    if getattr(provider_cls, "_SINGLE_CONNECTION", False):
        return f"model_lock:{entry.provider}"
    return f"model_lock:{entry.provider}/{entry.model}"


@asynccontextmanager
async def model_lock(
    pool: AsyncConnectionPool[Any], key: str, *, timeout_s: float
) -> AsyncIterator[bool]:
    """Hold *key*'s lock. Yields False if another job still has it after *timeout_s*."""
    async with pool.connection() as conn:
        try:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('lock_timeout', %s, true)",
                    (f"{round(timeout_s * 1000)}ms",),
                )
                await conn.execute(_ACQUIRE, (key,))
        except psycopg.errors.LockNotAvailable:
            held = False
        else:
            held = True
        try:
            yield held
        finally:
            if held:
                await _release(conn, key)


@asynccontextmanager
async def model_claim(pool: AsyncConnectionPool[Any], key: str) -> AsyncIterator[bool]:
    """Claim *key* for as long as the block runs. Yields False at once if another job has it.

    A claim is separate from the model lock. The lock is held for one provider call, so a
    probe can slip in between a sweep's calls. A claim is held for a whole model's worth
    of work, so two sweep runs never work the same model's queue.
    """
    claim = f"claim:{key}"
    async with pool.connection() as conn:
        async with conn.transaction():
            row = await (await conn.execute(_TRY_ACQUIRE, (claim,))).fetchone()
        held = bool(row["held"])
        try:
            yield held
        finally:
            if held:
                await _release(conn, claim)


async def _release(conn: Any, key: str) -> None:  # noqa: ANN401
    try:
        async with conn.transaction():
            await conn.execute(_RELEASE, (key,))
    except BaseException:
        await conn.close()
        raise
