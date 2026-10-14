# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._stream import run_stream


@pytest.mark.asyncio
async def test_cancelling_the_caller_cancels_both_tasks() -> None:
    started: list[str] = []
    cancelled: list[str] = []

    async def hang(name: str) -> None:
        started.append(name)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    caller = asyncio.create_task(
        run_stream(TranscriptionResult(provider="p"), hang("send"), hang("recv"))
    )
    while len(started) < 2:
        await asyncio.sleep(0)
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller
    assert sorted(cancelled) == ["recv", "send"]
