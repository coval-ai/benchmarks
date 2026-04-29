# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""WAV decode and resample helpers.

All benchmark audio is stored as 16 kHz mono PCM_16 WAV files (ADR-020).
These helpers are used by the dataset loader and STT providers.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict


class AudioMetadata(BaseModel):
    """Immutable metadata extracted from a WAV file."""

    model_config = ConfigDict(frozen=True)

    channels: int
    sample_width_bytes: int
    sample_rate: int
    duration_seconds: float


def load_wav(path: Path) -> tuple[bytes, AudioMetadata]:
    """Read a WAV file and return raw PCM bytes plus metadata.

    The returned ``bytes`` object is the raw PCM payload (everything after
    the 44-byte WAV header).  Pass these bytes directly to provider APIs
    that accept raw PCM audio.

    Args:
        path: Absolute path to a WAV file.

    Returns:
        A 2-tuple of ``(pcm_bytes, metadata)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If *path* is not a valid WAV file.
    """
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")

    data, sample_rate = sf.read(str(path), dtype="int16", always_2d=False)

    channels = 1 if data.ndim == 1 else data.shape[1]

    sample_width_bytes = 2  # int16 → 2 bytes per sample
    n_samples = data.shape[0]
    duration_seconds = n_samples / sample_rate

    metadata = AudioMetadata(
        channels=channels,
        sample_width_bytes=sample_width_bytes,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )

    pcm_bytes = data.tobytes()
    return pcm_bytes, metadata


def resample_to_16k(audio_array: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample *audio_array* to 16 000 Hz.

    Args:
        audio_array: A 1-D float32 numpy array of audio samples.
        src_rate: The sample rate of *audio_array*.

    Returns:
        A 1-D float32 numpy array resampled to 16 000 Hz.
    """
    if src_rate == 16_000:
        return audio_array

    resampled: np.ndarray = librosa.resample(
        audio_array,
        orig_sr=src_rate,
        target_sr=16_000,
    )
    return resampled


_WAV_HEADER_BYTES = 44


def to_pcm16_bytes(audio_array: np.ndarray, sample_rate: int = 16_000) -> bytes:
    """Convert a float32 audio array to raw 16-bit PCM bytes.

    Writes the array to an in-memory WAV container via soundfile, then strips
    the 44-byte WAV header to return only the PCM payload.

    Args:
        audio_array: A 1-D float32 numpy array of audio samples.
        sample_rate: The sample rate of *audio_array* (default 16 000 Hz).

    Returns:
        Raw 16-bit PCM bytes (no WAV header).
    """
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, subtype="PCM_16", format="WAV")
    buf.seek(0)
    wav_bytes = buf.read()
    return wav_bytes[_WAV_HEADER_BYTES:]
