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
                try:
                    async with conn.transaction():
                        await conn.execute(_RELEASE, (key,))
                except BaseException:
                    await conn.close()
                    raise
