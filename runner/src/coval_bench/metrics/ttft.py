# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Time-To-First-Token (TTFT), audio-to-final, and Real-Time Factor metrics for STT providers."""

from __future__ import annotations


def compute_ttft(audio_send_start: float, first_token_at: float) -> float:
    """Return time-to-first-token in **seconds**.

    Both arguments must be ``time.monotonic()`` readings.

    Args:
        audio_send_start: Monotonic timestamp immediately before audio is sent
            to the STT provider.
        first_token_at: Monotonic timestamp when the first transcript token is
            received.

    Returns:
        Elapsed time in seconds (float).

    Raises:
        ValueError: If *first_token_at* is earlier than *audio_send_start*.
    """
    if first_token_at < audio_send_start:
        raise ValueError("first_token_at must be >= audio_send_start")
    return first_token_at - audio_send_start


def compute_audio_to_final(audio_send_start: float, final_transcript_at: float) -> float:
    """Return audio-to-final-transcript latency in **seconds**.

    Args:
        audio_send_start: Monotonic timestamp immediately before audio is sent.
        final_transcript_at: Monotonic timestamp when the final (complete)
            transcript is received.

    Returns:
        Elapsed time in seconds (float).

    Raises:
        ValueError: If *final_transcript_at* is earlier than *audio_send_start*.
    """
    if final_transcript_at < audio_send_start:
        raise ValueError("final_transcript_at must be >= audio_send_start")
    return final_transcript_at - audio_send_start


def compute_rtf(audio_to_final_seconds: float, audio_duration_seconds: float) -> float:
    """Return Real-Time Factor (RTF).

    RTF = wall-clock processing time / audio duration.
    Values < 1.0 indicate faster-than-real-time transcription.

    Args:
        audio_to_final_seconds: Wall-clock seconds from audio send to final
            transcript (output of :func:`compute_audio_to_final`).
        audio_duration_seconds: Duration of the audio clip in seconds.

    Returns:
        RTF as a dimensionless ratio.

    Raises:
        ValueError: If *audio_duration_seconds* is not positive.
    """
    if audio_duration_seconds <= 0:
        raise ValueError("audio_duration_seconds must be > 0")
    return audio_to_final_seconds / audio_duration_seconds
