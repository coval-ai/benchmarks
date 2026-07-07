# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt._pacing.

The clock and sleep are patched so the schedule is asserted deterministically
without real waiting. The key invariant: every chunk is paced, including the
last, so the end-of-stream signal lands at the true end of the audio.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from coval_bench.providers.stt import _pacing

_BYTE_RATE = 32000  # 16 kHz * 2 bytes, mono


def test_pacing_delay_positive_when_ahead(monkeypatch: pytest.MonkeyPatch) -> None:
    # 16000 bytes at 32000 B/s == 0.5 s of audio; deadline start+0.5, now == start.
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    assert _pacing.pacing_delay(10.0, 16000, _BYTE_RATE) == pytest.approx(0.5)


def test_pacing_delay_nonpositive_when_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    assert _pacing.pacing_delay(10.0, 16000, _BYTE_RATE) < 0


async def test_paces_every_chunk_including_last(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    audio = b"\x00" * 1000
    yielded = [pair async for pair in _pacing.paced_chunks(audio, 300, _BYTE_RATE)]
    chunks = [chunk for chunk, _ in yielded]

    # Chunks tile the whole clip, last one short.
    assert b"".join(chunks) == audio
    assert [len(c) for c in chunks] == [300, 300, 300, 100]
    # Every chunk is paced — the last included (skip-last would give 3).
    assert len(sleeps) == len(chunks) == 4
    # The final deadline targets the true audio duration (1000 bytes).
    assert sleeps[-1] == pytest.approx(1000 / _BYTE_RATE)
    assert yielded[-1][1] == 1000.0


async def test_min_tail_bytes_merges_short_final(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    audio = b"\x00" * 1000
    chunks = [c async for c, _ in _pacing.paced_chunks(audio, 300, _BYTE_RATE, min_tail_bytes=150)]
    assert b"".join(chunks) == audio
    assert [len(c) for c in chunks] == [300, 300, 400]


async def test_min_tail_bytes_leaves_adequate_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    audio = b"\x00" * 1000
    chunks = [c async for c, _ in _pacing.paced_chunks(audio, 300, _BYTE_RATE, min_tail_bytes=100)]
    assert [len(c) for c in chunks] == [300, 300, 300, 100]


async def test_min_tail_bytes_single_chunk_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    audio = b"\x00" * 50
    chunks = [c async for c, _ in _pacing.paced_chunks(audio, 300, _BYTE_RATE, min_tail_bytes=150)]
    assert [len(c) for c in chunks] == [50]


def test_sync_min_tail_bytes_merges_short_final(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(time, "sleep", lambda delay: None)

    chunks = [
        c for c, _ in _pacing.paced_chunks_sync(b"\x00" * 500, 200, _BYTE_RATE, min_tail_bytes=150)
    ]
    assert [len(c) for c in chunks] == [200, 300]


async def test_chunk_size_floored_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    # chunk_size 0 would hang a naive loop; the helper floors it to 1.
    chunks = [chunk async for chunk, _ in _pacing.paced_chunks(b"abc", 0, _BYTE_RATE)]
    assert b"".join(chunks) == b"abc"
    assert [len(c) for c in chunks] == [1, 1, 1]


async def test_respects_start_override(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(time, "monotonic", lambda: 5.0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    starts = [s async for _, s in _pacing.paced_chunks(b"abcd", 2, _BYTE_RATE, start=42.0)]
    assert starts == [42.0, 42.0]


def test_sync_paces_every_chunk_including_last(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(time, "sleep", lambda delay: sleeps.append(delay))

    chunks = [chunk for chunk, _ in _pacing.paced_chunks_sync(b"\x00" * 500, 200, _BYTE_RATE)]
    assert [len(c) for c in chunks] == [200, 200, 100]
    assert len(sleeps) == len(chunks) == 3
    assert sleeps[-1] == pytest.approx(500 / _BYTE_RATE)
