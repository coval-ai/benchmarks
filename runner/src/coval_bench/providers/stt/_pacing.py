# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-time pacing for streaming STT audio sends.

Providers are fed audio at 1x real time so latency metrics reflect provider
processing rather than feed timing. Each chunk is paced to an absolute deadline
(``start + cumulative_bytes / byte_rate``), and the final chunk is paced too so
the end-of-stream signal lands at the true end of the audio.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator


def pacing_delay(start: float, sent_bytes: int, byte_rate: int) -> float:
    """Seconds to wait so ``sent_bytes`` of audio matches real time.

    Non-positive when the send is already behind schedule (no wait needed).
    """
    return start + sent_bytes / byte_rate - time.monotonic()


def _chunk_spans(total: int, chunk_size: int, min_tail_bytes: int) -> list[tuple[int, int]]:
    spans = [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]
    if min_tail_bytes and len(spans) >= 2:
        last_lo, last_hi = spans[-1]
        if last_hi - last_lo < min_tail_bytes:
            prev_lo, _ = spans.pop(-2)
            spans[-1] = (prev_lo, last_hi)
    return spans


async def paced_chunks(
    audio_data: bytes,
    chunk_size: int,
    byte_rate: int,
    *,
    start: float | None = None,
    min_tail_bytes: int = 0,
) -> AsyncIterator[tuple[bytes, float]]:
    """Yield ``(chunk, start)`` pacing each chunk to real time, the last included.

    ``start`` defaults to the monotonic clock at the first chunk; pass it to
    anchor pacing to a timestamp captured earlier. ``min_tail_bytes`` folds a
    sub-threshold final chunk into its predecessor.
    """
    chunk_size = max(1, chunk_size)
    if start is None:
        start = time.monotonic()
    sent_bytes = 0
    for lo, hi in _chunk_spans(len(audio_data), chunk_size, min_tail_bytes):
        yield audio_data[lo:hi], start
        sent_bytes += hi - lo
        delay = pacing_delay(start, sent_bytes, byte_rate)
        if delay > 0:
            await asyncio.sleep(delay)


def paced_chunks_sync(
    audio_data: bytes,
    chunk_size: int,
    byte_rate: int,
    *,
    start: float | None = None,
    min_tail_bytes: int = 0,
) -> Iterator[tuple[bytes, float]]:
    """Synchronous :func:`paced_chunks` for blocking send loops (e.g. gRPC)."""
    chunk_size = max(1, chunk_size)
    if start is None:
        start = time.monotonic()
    sent_bytes = 0
    for lo, hi in _chunk_spans(len(audio_data), chunk_size, min_tail_bytes):
        yield audio_data[lo:hi], start
        sent_bytes += hi - lo
        delay = pacing_delay(start, sent_bytes, byte_rate)
        if delay > 0:
            time.sleep(delay)
