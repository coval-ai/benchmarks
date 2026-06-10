# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process TTL cache for read-only dashboard endpoints.

``/v1/results/aggregates`` recomputes the same SQL on every request even though
the underlying results only change when a benchmark run finishes (~every 30
min). A short per-process TTL cache lets identical requests inside the window
skip the DB.

Tradeoffs: lost on restart, not shared across API instances, and the first
request after expiry still scans raw. A shared/durable cache would be Redis —
a deliberately larger call deferred for now.

One cache lives per app instance (see ``create_app``). Keys are namespaced
tuples so the cache can be shared by more endpoints later without collisions;
only ``/v1/results/aggregates`` reads it today.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Hashable
from typing import Any, Literal

from cachetools import TTLCache

# 15 min cache for no real reason.
CACHE_TTL_SECONDS = 900
# How long one fill failure is shared before the query is retried.
FAILURE_TTL_SECONDS = 5

CacheStatus = Literal["hit", "coalesced", "miss"]


def new_response_cache() -> TTLCache[Any, Any]:
    """Build the per-app response cache.

    ``maxsize`` is ample headroom: the only keys are the few (benchmark,
    window) aggregates param combinations.
    """
    return TTLCache(maxsize=64, ttl=CACHE_TTL_SECONDS)


def new_cache_locks() -> defaultdict[Any, asyncio.Lock]:
    """Per-cache-key locks backing ``get_or_fill``."""
    return defaultdict(asyncio.Lock)


async def get_or_fill[T](
    cache: TTLCache[Any, Any],
    locks: defaultdict[Any, asyncio.Lock],
    key: Hashable,
    fill: Callable[[], Awaitable[T]],
) -> tuple[T, CacheStatus]:
    """Read ``key`` from the cache, running ``fill`` once on a miss.

    Concurrent misses for one key share a single ``fill`` instead of each
    holding a pool connection ("coalesced"). A fill that raises is re-raised
    to callers arriving within FAILURE_TTL_SECONDS, so a failing query is
    not re-run serially by every queued waiter.
    """
    lock = locks[key]
    waited = lock.locked()
    async with lock:
        value = cache.get(key)
        if value is not None:
            return value, "coalesced" if waited else "hit"

        failure = cache.get((key, "failure"))
        if failure is not None:
            failed_at, exc = failure
            if time.monotonic() - failed_at < FAILURE_TTL_SECONDS:
                raise exc

        try:
            value = await fill()
        except Exception as exc:
            cache[(key, "failure")] = (time.monotonic(), exc)
            raise
        cache[key] = value
        return value, "miss"
