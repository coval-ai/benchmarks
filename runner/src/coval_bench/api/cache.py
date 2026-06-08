# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process TTL cache for read-only dashboard endpoints.

``/v1/results/aggregates`` and ``/v1/leaderboard`` recompute the same SQL on
every request even though the underlying results only change when a benchmark
run finishes (~every 30 min). A short per-process TTL cache lets identical
requests inside the window skip the DB.

Tradeoffs: lost on restart, not shared across API instances, and the first
request after expiry still scans raw. A shared/durable cache would be Redis —
a deliberately larger call deferred for now.

One cache lives per app instance (see ``create_app``); both endpoints share it
with namespaced tuple keys.
"""

from __future__ import annotations

from typing import Any

from cachetools import TTLCache

# 15 min cache for no real reason.
CACHE_TTL_SECONDS = 900


def new_response_cache() -> TTLCache[Any, Any]:
    """Build the per-app response cache.

    ``maxsize`` is ample headroom: the only keys are the few (benchmark,
    window) and (metric, benchmark, window) param combinations.
    """
    return TTLCache(maxsize=64, ttl=CACHE_TTL_SECONDS)
