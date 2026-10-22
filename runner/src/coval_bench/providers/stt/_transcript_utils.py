# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared transcript helpers for streaming STT providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from coval_bench.providers.base import TranscriptionResult

PartialFallback = Literal["last", "longest", "none"]


def set_first_token(
    result: TranscriptionResult,
    transcript: str,
    *,
    now: float,
    max_chars: int = 30,
) -> None:
    """Populate TTFT and first-token preview from the first non-empty transcript."""
    clean = transcript.strip()
    if not clean or result.ttft_seconds is not None or result.audio_start_time is None:
        return

    result.ttft_seconds = now - result.audio_start_time
    result.first_token_content = clean[:max_chars] + "..." if len(clean) > max_chars else clean


def add_partial_transcript(result: TranscriptionResult, transcript: str) -> None:
    """Append a non-empty partial transcript snapshot."""
    clean = transcript.strip()
    if clean:
        result.partial_transcripts.append(clean)


def finalize_transcript(
    result: TranscriptionResult,
    *,
    explicit_transcript: str | None = None,
    final_segments: Sequence[str] | None = None,
    partial_fallback: PartialFallback = "longest",
    join: str = " ",
) -> None:
    """Set ``complete_transcript`` and derived stats with a safe fallback policy.

    ``join`` is the separator between final segments; providers whose segments
    carry their own leading spaces pass ``""``. With ``partial_fallback="none"``
    a clip without a final keeps ``complete_transcript`` None so WER scores it
    as failed rather than from a fragment.
    """
    transcript: str | None = None

    if explicit_transcript is not None and explicit_transcript.strip():
        transcript = explicit_transcript.strip()
    else:
        if join:
            cleaned_finals = [seg.strip() for seg in final_segments or [] if seg.strip()]
        else:
            cleaned_finals = [seg for seg in final_segments or [] if seg]
        if cleaned_finals:
            transcript = join.join(cleaned_finals).strip() or None
        elif partial_fallback != "none":
            cleaned_partials = [
                partial.strip() for partial in result.partial_transcripts if partial.strip()
            ]
            if cleaned_partials:
                if partial_fallback == "last":
                    transcript = cleaned_partials[-1]
                else:
                    transcript = max(cleaned_partials, key=len)

    result.complete_transcript = transcript
    if transcript:
        result.transcript_length = len(transcript)
        result.word_count = len(transcript.split())
