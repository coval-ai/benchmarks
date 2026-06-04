# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared transcript helpers for streaming STT providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from coval_bench.providers.base import TranscriptionResult


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
    partial_fallback: Literal["last", "longest"] = "longest",
) -> None:
    """Set ``complete_transcript`` and derived stats with a safe fallback policy."""
    transcript: str | None = None

    if explicit_transcript is not None and explicit_transcript.strip():
        transcript = explicit_transcript.strip()
    else:
        cleaned_finals = [segment.strip() for segment in final_segments or [] if segment.strip()]
        if cleaned_finals:
            transcript = " ".join(cleaned_finals).strip() or None
        else:
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
