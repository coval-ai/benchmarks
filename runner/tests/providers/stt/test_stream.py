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


@pytest.mark.asyncio
async def test_receiver_failure_that_cancels_an_unfinished_send_fails_the_clip() -> None:
    """A socket dying mid-clip must not score as a success because an early final landed."""
    result = TranscriptionResult(provider="p", audio_start_time=0.0, audio_to_final_seconds=1.0)

    async def send_forever() -> None:
        await asyncio.Event().wait()

    async def recv_dies() -> None:
        await asyncio.sleep(0)
        raise ConnectionError("socket dropped")

    await run_stream(result, send_forever(), recv_dies())

    assert result.error == "socket dropped"


@pytest.mark.asyncio
async def test_receiver_failure_after_a_completed_send_keeps_an_earlier_final() -> None:
    result = TranscriptionResult(provider="p", audio_start_time=0.0, audio_to_final_seconds=1.0)

    async def send_done() -> None:
        return None

    async def recv_dies() -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        raise ConnectionError("late close")

    await run_stream(result, send_done(), recv_dies())

    assert result.error is None
