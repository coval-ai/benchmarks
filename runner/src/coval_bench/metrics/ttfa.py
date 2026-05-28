# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Time-To-First-Audio (TTFA) metric for TTS providers.

Perceived TTFA is the wall-clock time until the first audio chunk arrives plus
the leading silence inside the stream before the first audible sample, which is
what an enqueue-and-play client actually experiences.
"""

from __future__ import annotations

import librosa
import numpy as np

# Loudness threshold and framing calibrated against recorded output from every
# enabled provider. The threshold is an RMS level on normalized float audio
# (range [-1, 1]); 0.01 sits above the worst leading noise floor and below
# every voiced onset. Frame/hop are time-based so the window is a true 10 ms
# regardless of sample rate (providers range 22.05k-48k).
_AUDIBLE_RMS_THRESHOLD = 0.01
_FRAME_SECONDS = 0.010
_HOP_SECONDS = 0.001

# Signed 16-bit PCM: 2 bytes per sample, full-scale 2**15 to normalize to [-1, 1].
_PCM16_BYTES = 2
_PCM16_FULL_SCALE = 32768.0


def first_audible_offset_ms(
    pcm: bytes,
    sample_rate: int,
) -> float | None:
    """Offset in milliseconds from the start of *pcm* to the first audible sample.

    *pcm* is assembled raw mono PCM (no WAV header), signed little-endian 16-bit,
    interpreted at its native *sample_rate* (no resampling). Returns ``None``
    when there is no audible audio (empty, pure silence, or sub-threshold).
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    if not pcm:
        return None
    if len(pcm) % _PCM16_BYTES != 0:
        raise ValueError("pcm length is not a whole number of frames")

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _PCM16_FULL_SCALE
    if samples.size == 0:
        return None

    frame_length = max(1, round(_FRAME_SECONDS * sample_rate))
    hop_length = max(1, round(_HOP_SECONDS * sample_rate))

    # Per-frame RMS with each frame's mean removed, so a (silent) DC offset
    # doesn't read as energy and trip the threshold. Edge-pad rather than
    # zero-pad: a zero-pad would make a constant DC start look like a step,
    # which has real energy and would falsely read as audible at t=0.
    pad = frame_length // 2
    padded = np.pad(samples, pad, mode="edge")
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean((frames - frames.mean(axis=0, keepdims=True)) ** 2, axis=0))

    audible_frames = np.where(rms > _AUDIBLE_RMS_THRESHOLD)[0]
    if audible_frames.size == 0:
        return None

    return (float(audible_frames[0]) * hop_length / sample_rate) * 1000.0


def compute_ttfa(
    audio_synthesis_start: float,
    first_audio_chunk_at: float,
    leading_silence_offset_ms: float = 0.0,
) -> float:
    """Return perceived time-to-first-audio in milliseconds.

    Perceived TTFA = (first chunk arrival - synthesis start) +
    *leading_silence_offset_ms*. The timestamps are ``time.monotonic()``
    readings; the offset is the output of :func:`first_audible_offset_ms`.
    """
    if first_audio_chunk_at < audio_synthesis_start:
        raise ValueError("first_audio_chunk_at must be >= audio_synthesis_start")
    if leading_silence_offset_ms < 0:
        raise ValueError("leading_silence_offset_ms must be >= 0")
    arrival_ms = (first_audio_chunk_at - audio_synthesis_start) * 1000.0
    return arrival_ms + leading_silence_offset_ms
