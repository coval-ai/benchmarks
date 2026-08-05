# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared TTS finalize helper (perceived TTFA + WAV output)."""

from __future__ import annotations

import math
import wave
from unittest.mock import patch

import numpy as np
import pytest

from coval_bench.providers.tts._common import finalize_tts_result


def _silence_pcm(duration_ms: float, sample_rate: int) -> bytes:
    """Return *duration_ms* of pure silence as int16 PCM bytes."""
    n = round(duration_ms / 1000.0 * sample_rate)
    return np.zeros(n, dtype="<i2").tobytes()


def _tone_pcm(
    duration_ms: float, sample_rate: int, amplitude: float = 0.3, freq: float = 220.0
) -> bytes:
    """Return a *duration_ms* sine tone as int16 PCM bytes (amplitude in [0, 1])."""
    n = round(duration_ms / 1000.0 * sample_rate)
    t = np.arange(n) / sample_rate
    wave_arr = amplitude * np.sin(2.0 * math.pi * freq * t)
    return (wave_arr * 32767.0).astype("<i2").tobytes()


def test_finalize_adds_leading_silence_offset() -> None:
    """Silence-then-tone PCM → ttfa = arrival + offset, strictly greater than arrival."""
    sr = 24000
    lead_ms = 200.0
    pcm = _silence_pcm(lead_ms, sr) + _tone_pcm(300, sr)
    arrival_ms = 500.0

    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=pcm,
        sample_rate=sr,
        audio_synthesis_start=1000.0,
        first_audio_chunk_at=1000.0 + arrival_ms / 1000.0,
    )

    assert result.ttfa_ms is not None
    assert result.ttfa_ms > arrival_ms
    assert result.ttfa_ms == pytest.approx(arrival_ms + lead_ms, abs=12.0)
    assert result.leading_silence_ms is not None
    assert result.ttfa_ms == arrival_ms + result.leading_silence_ms

    assert result.audio_path is not None
    assert result.audio_path.exists()
    with wave.open(str(result.audio_path), "rb") as wav_file:
        assert wav_file.getframerate() == sr
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
    result.audio_path.unlink()


def test_finalize_immediate_tone_offset_near_zero() -> None:
    """Immediate-tone PCM → ttfa ≈ arrival (offset ≈ 0)."""
    sr = 24000
    arrival_ms = 300.0
    pcm = _tone_pcm(300, sr)

    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=pcm,
        sample_rate=sr,
        audio_synthesis_start=2.0,
        first_audio_chunk_at=2.0 + arrival_ms / 1000.0,
    )

    assert result.ttfa_ms is not None
    assert result.ttfa_ms == pytest.approx(arrival_ms, abs=10.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_finalize_error_path_arrival_only() -> None:
    """Error path (empty pcm, timestamps set) → ttfa = arrival only, no WAV."""
    arrival_ms = 420.0
    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=b"",
        sample_rate=24000,
        audio_synthesis_start=10.0,
        first_audio_chunk_at=10.0 + arrival_ms / 1000.0,
        error="boom",
    )

    assert result.ttfa_ms == pytest.approx(arrival_ms)
    assert result.audio_path is None
    assert result.error == "boom"


def test_finalize_odd_length_pcm_still_writes_wav() -> None:
    """A dangling partial sample (split final frame) is dropped, not fatal.

    Regression: offset detection rejects frame-misaligned PCM. The finalize step
    must align it, still detect the leading-silence offset, and write the WAV.
    """
    sr = 24000
    lead_ms = 200.0
    pcm = _silence_pcm(lead_ms, sr) + _tone_pcm(300, sr) + b"\x07"  # stray odd byte

    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=pcm,
        sample_rate=sr,
        audio_synthesis_start=1.0,
        first_audio_chunk_at=1.5,
    )

    assert result.ttfa_ms is not None
    assert result.ttfa_ms == pytest.approx(500.0 + lead_ms, abs=12.0)
    assert result.audio_path is not None
    assert result.audio_path.exists()
    with wave.open(str(result.audio_path), "rb") as wav_file:
        # The dangling byte was dropped → frame count is a whole number of samples.
        assert wav_file.getnframes() == (len(pcm) - 1) // 2
    result.audio_path.unlink()


def test_finalize_offset_failure_falls_back_and_writes_wav() -> None:
    """If offset detection raises, TTFA degrades to arrival-only and the WAV still writes.

    A latency-metric failure must never discard good synthesized audio.
    """
    sr = 24000
    pcm = _tone_pcm(300, sr)
    arrival_ms = 250.0

    with patch(
        "coval_bench.providers.tts._common.first_audible_offset_ms",
        side_effect=RuntimeError("offset boom"),
    ):
        result = finalize_tts_result(
            provider="test",
            model="m",
            voice="v",
            pcm=pcm,
            sample_rate=sr,
            audio_synthesis_start=10.0,
            first_audio_chunk_at=10.0 + arrival_ms / 1000.0,
        )

    assert result.ttfa_ms == pytest.approx(arrival_ms)  # arrival only, offset dropped
    assert result.leading_silence_ms is None  # unknown split, not 0.0
    assert result.error is None
    assert result.audio_path is not None
    assert result.audio_path.exists()
    result.audio_path.unlink()


def test_finalize_no_audio_returns_none() -> None:
    """No audio (first_audio_chunk_at unset) → ttfa None, no WAV, and a stamped reason.

    Empty pcm with no error is a silent failure, not a success: the provider's receive
    loop ended without synthesising anything and without recognising why.
    """
    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=b"",
        sample_rate=24000,
        audio_synthesis_start=10.0,
        first_audio_chunk_at=None,
    )

    assert result.ttfa_ms is None
    assert result.audio_path is None
    assert result.error == "provider closed the stream without sending audio or an error"


def test_finalize_silent_failure_quotes_last_frames() -> None:
    """Retained frames are appended verbatim, so an unmatched schema still explains itself.

    This is the Hume case: it sent a plain-English billing message under a key the
    integration didn't match. The wording must survive without provider-specific code.
    """
    hume_frame = (
        '{"status_code":400,"message":"Exhausted credit balance.",'
        '"details":{"type":"error","code":"E0300","slug":"zero_credits"}}'
    )
    result = finalize_tts_result(
        provider="hume",
        model="octave-2",
        voice="v",
        pcm=b"",
        sample_rate=24000,
        audio_synthesis_start=10.0,
        first_audio_chunk_at=None,
        last_frames=[hume_frame],
    )

    assert result.error is not None
    assert result.error.startswith("provider closed the stream without sending audio or an error")
    assert "Exhausted credit balance." in result.error


def test_finalize_does_not_override_a_reported_error() -> None:
    """A provider that did report a reason keeps its own wording."""
    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=b"",
        sample_rate=24000,
        audio_synthesis_start=None,
        first_audio_chunk_at=None,
        error="rate limited",
        last_frames=["ignored"],
    )

    assert result.error == "rate limited"


def test_finalize_successful_audio_is_untouched() -> None:
    """Audio present → the guard stays out of the way."""
    sr = 24000
    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=_tone_pcm(200, sr),
        sample_rate=sr,
        audio_synthesis_start=10.0,
        first_audio_chunk_at=10.2,
    )

    assert result.error is None
    assert result.audio_path is not None


def test_finalize_inaudible_audio_fails_instead_of_reporting_ttfa() -> None:
    """Silence-only audio is a synthesis failure, not a fast TTFA.

    With no audible frame the offset is null, which used to collapse to 0.0 and make
    TTFA pure arrival time — a plausible number measured off silence that then entered
    the aggregates.
    """
    sr = 24000
    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=_silence_pcm(500, sr),
        sample_rate=sr,
        audio_synthesis_start=10.0,
        first_audio_chunk_at=10.3,
    )

    assert result.error == "provider audio remained below the audibility threshold"
    assert result.ttfa_ms is None


def test_finalize_offset_detection_crash_does_not_condemn_the_audio() -> None:
    """A crash in offset detection is our problem — keep the audio and the TTFA.

    Distinct from genuine silence: both yield a null offset, and conflating them would
    fail perfectly good audio whenever our own detection threw.
    """
    sr = 24000
    with patch(
        "coval_bench.providers.tts._common.first_audible_offset_ms",
        side_effect=RuntimeError("detector exploded"),
    ):
        result = finalize_tts_result(
            provider="test",
            model="m",
            voice="v",
            pcm=_tone_pcm(300, sr),
            sample_rate=sr,
            audio_synthesis_start=10.0,
            first_audio_chunk_at=10.3,
        )

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(300.0)
    assert result.audio_path is not None
    assert result.audio_path.exists()
    result.audio_path.unlink()
