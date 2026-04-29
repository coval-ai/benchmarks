# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Time-To-First-Audio (TTFA) metric for TTS providers."""

from __future__ import annotations


def compute_ttfa(audio_synthesis_start: float, first_audio_chunk_at: float) -> float:
    """Return time-to-first-audio in **milliseconds**.

    Both arguments must be ``time.monotonic()`` readings captured by the
    orchestrator around the TTS request.

    Args:
        audio_synthesis_start: Monotonic timestamp immediately before the TTS
            request is issued.
        first_audio_chunk_at: Monotonic timestamp when the first audio chunk
            is received from the provider.

    Returns:
        Elapsed time in milliseconds (float).

    Raises:
        ValueError: If *first_audio_chunk_at* is earlier than
            *audio_synthesis_start*.
    """
    if first_audio_chunk_at < audio_synthesis_start:
        raise ValueError("first_audio_chunk_at must be >= audio_synthesis_start")
    return (first_audio_chunk_at - audio_synthesis_start) * 1000.0
