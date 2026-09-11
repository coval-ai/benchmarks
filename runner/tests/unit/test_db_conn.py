# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared connection pool lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from coval_bench.db import conn


class _Pool:
    instances: list[_Pool] = []

    def __init__(self, **_: Any) -> None:
        self.opened = 0
        self.closed = True
        self.instances.append(self)

    async def open(self) -> None:
        if self.opened:
            raise RuntimeError("cannot reuse")
        self.opened += 1
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_lifespan_pool_can_be_entered_again_after_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conn, "AsyncConnectionPool", _Pool)
    monkeypatch.setattr(conn, "_pool", None)
    settings = SimpleNamespace(database_url="postgresql://x@127.0.0.1:1/x")

    async with conn.lifespan_pool(settings) as first:  # type: ignore[arg-type]
        assert not first.closed
    async with conn.lifespan_pool(settings) as second:  # type: ignore[arg-type]
        assert not second.closed
    assert first is not second
    assert first.closed and second.closed
