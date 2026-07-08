# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Real-Time Factor and Time-To-Final-Segment metrics for STT providers."""

from __future__ import annotations


def compute_rtf(audio_to_final_seconds: float, audio_duration_seconds: float) -> float:
    """Return Real-Time Factor (RTF).

    RTF = wall-clock processing time / audio duration.
    Values < 1.0 indicate faster-than-real-time transcription.

    Args:
        audio_to_final_seconds: Wall-clock seconds from audio send to final
            transcript.
        audio_duration_seconds: Duration of the audio clip in seconds.

    Returns:
        RTF as a dimensionless ratio.

    Raises:
        ValueError: If *audio_duration_seconds* is not positive.
    """
    if audio_duration_seconds <= 0:
        raise ValueError("audio_duration_seconds must be > 0")
    return audio_to_final_seconds / audio_duration_seconds


def compute_ttfs(audio_to_final_seconds: float, speech_end_offset_seconds: float) -> float:
    """Return Time-To-Final-Segment (TTFS) in **seconds**.

    TTFS is latency from VAD-detected end-of-speech to the final transcript:
    ``audio_to_final_seconds - speech_end_offset_seconds``. The offset is the
    precomputed, shared end-of-speech anchor for the clip (same for every
    provider), so TTFS isolates finalization speed given a client end-of-speech
    signal.

    Args:
        audio_to_final_seconds: Wall-clock seconds from audio send to final transcript.
        speech_end_offset_seconds: VAD end-of-speech offset for the clip.

    Returns:
        Elapsed time in seconds, clamped at 0.0 for finals ahead of the anchor.
    """
    return max(0.0, audio_to_final_seconds - speech_end_offset_seconds)
