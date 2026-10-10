# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Send/receive task orchestration shared by streaming STT providers."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from coval_bench.providers.base import TranscriptionResult


async def run_stream(
    result: TranscriptionResult,
    send: Coroutine[Any, Any, object],
    recv: Coroutine[Any, Any, object],
    *,
    no_final_error: str | None = None,
) -> None:
    """Run the sender and receiver together, recording the first failure on *result*.

    A raised exception cancels the other task. A sender failure always fails the
    clip, since a socket that drops mid-clip leaves the transcript cut off even
    after an earlier final. A receiver failure is only stamped when no final
    landed. ``no_final_error`` is stamped when both tasks finish cleanly without
    a final; stamping it here rather than in a receiver keeps a mid-stream close
    from masking a sender exception. Cancelling the caller cancels both tasks.
    """
    tasks = (asyncio.create_task(send), asyncio.create_task(recv))
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        if any(not task.cancelled() and task.exception() is not None for task in done):
            for task in pending:
                task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if result.error is None and isinstance(outcomes[0], Exception):
        result.error = str(outcomes[0])
    if result.error is None and result.audio_to_final_seconds is None:
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                result.error = str(outcome)
                break
        else:
            if no_final_error is not None:
                result.error = no_final_error
