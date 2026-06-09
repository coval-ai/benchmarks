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
from collections import defaultdict
from typing import Any

from cachetools import TTLCache

# 15 min cache for no real reason.
CACHE_TTL_SECONDS = 900


def new_response_cache() -> TTLCache[Any, Any]:
    """Build the per-app response cache.

    ``maxsize`` is ample headroom: the only keys are the few (benchmark,
    window) aggregates param combinations.
    """
    return TTLCache(maxsize=64, ttl=CACHE_TTL_SECONDS)


def new_cache_locks() -> defaultdict[Any, asyncio.Lock]:
    """Per-cache-key locks that coalesce concurrent misses.

    N simultaneous requests for one uncached key run the SQL once; the rest
    wait on the lock instead of each holding one of the pool's 4 connections.
    """
    return defaultdict(asyncio.Lock)
