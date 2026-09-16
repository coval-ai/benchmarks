"""Per-model concurrency gate for provider requests.

Every registered model gets its own lock, so a model never serves two of our
requests at once and its latency numbers never include self-contention.
Different models proceed side by side, bounded only by a global cap on open
provider connections.

A task takes its model lock before waiting on the global slot. Holding the lock
while queued blocks only siblings of the same model, which could not run anyway,
and keeps the global slots for tasks that can actually make progress.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager


class ModelGate:
    def __init__(self, cap: int) -> None:
        if cap < 1:
            raise ValueError("cap must be at least 1")
        self.cap = cap
        self._global = asyncio.Semaphore(cap)
        self._locks: defaultdict[Hashable, asyncio.Lock] = defaultdict(asyncio.Lock)

    @asynccontextmanager
    async def slot(self, key: Hashable) -> AsyncIterator[None]:
        """Hold *key*'s lock and one global slot for the duration of a request."""
        async with self._locks[key], self._global:
            yield

    @asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        """Hold one global slot without pinning a model; for side work such as uploads."""
        async with self._global:
            yield
