# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face ``PolyAI/minds14`` (en-US ``train``) decode for ``build_dataset`` stt-v2.

HF revision pin (``HF_MINDS14_REVISION``)
-----------------------------------------
Hugging Face dataset cards are backed by a **git** repository snapshot. Passing
``revision=`` into ``datasets.load_dataset`` selects which commit/tag/branch Hub
should serve rows from. Different revisions can reorder rows, tweak metadata,
or even change decoding — so WAV bytes you rebuild today might not match SHA256
values locked in git unless you fetch the **same revision** that was used when
those SHAs were recorded.

Using ``branch`` names (e.g. ``main``) is weak: they float. Prefer the **dataset
repository commit SHA** cited on the HF dataset card (or pinned in CI) alongside
committed ``stt-v2.json``.

This module passes the env override through to Hub; **`stt-v2.json`` remains the
authority for ``path`` + ``sha256``**.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_HF_PATH = "PolyAI/minds14"
_HF_NAME = "en-US"
_HF_SPLIT = "train"
_TARGET_SR = 16_000


def hf_revision_env() -> str | None:
    """Return HF ``revision`` string, or ``None`` if unset/blank (Hub default snapshot)."""
    raw = os.environ.get("HF_MINDS14_REVISION", "").strip()
    return raw or None


def load_train_dataset(*, revision: str | None = None) -> Any:  # noqa: ANN401
    """Return the Hugging Face ``datasets`` Dataset for en-US train.

    The Hub schema uses an ``Audio`` column; recent ``datasets`` versions decode
    that column with ``torchcodec`` by default. We cast to ``Audio(decode=False)``
    so iteration yields ``path`` / ``bytes`` only and we decode in
    :func:`decode_audio` via ``soundfile`` + ``librosa`` (no PyTorch/torchcodec).
    """
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install HF build deps: cd runner && uv sync --extra minds14-build "
            "(or run with --extra minds14-build)"
        ) from exc

    rev = revision if revision is not None else hf_revision_env()
    load_kw: dict[str, Any] = {"path": _HF_PATH, "name": _HF_NAME, "split": _HF_SPLIT}
    if rev:
        load_kw["revision"] = rev
    ds = load_dataset(**load_kw)
    return ds.cast_column("audio", Audio(decode=False))


def collect_audio_by_path(wanted: set[str], *, revision: str | None = None) -> dict[str, Any]:
    """Iterate the train split; map each ``path`` in *wanted* to its ``audio`` column."""
    dataset = cast(Iterable[Mapping[str, Any]], load_train_dataset(revision=revision))
    found: dict[str, Any] = {}
    for row in dataset:
        p = str(row.get("path", "")).strip()
        if p not in wanted:
            continue
        found[p] = row.get("audio")

    missing = wanted - found.keys()
    if missing:
        ordered = sorted(missing)
        logger.warning("HF paths missing from split (%s): %s", len(missing), ordered)
        sample = ", ".join(ordered[:5])
        more = "" if len(missing) <= 5 else f" … (+{len(missing) - 5})"
        raise ValueError(
            f"{len(missing)} path(s) not found in {_HF_PATH} {_HF_SPLIT}: {sample}{more}"
        )

    return found


def decode_audio(audio_info: object) -> tuple[np.ndarray, int]:
    """Decode Hugging Face ``Audio`` payloads (typically ``{'array','sampling_rate'}`` dict)."""
    if isinstance(audio_info, dict):
        if "array" in audio_info and "sampling_rate" in audio_info:
            arr = np.asarray(audio_info["array"], dtype=np.float32)
            sr = int(audio_info["sampling_rate"])
            return arr, sr

        raw_bytes = audio_info.get("bytes")
        if raw_bytes:
            raw, sr_use = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            return np.asarray(raw, dtype=np.float32), int(sr_use)

        disk_path = audio_info.get("path")
        if disk_path:
            raw, sr_use = sf.read(str(disk_path), dtype="float32", always_2d=False)
            return np.asarray(raw, dtype=np.float32), int(sr_use)

        raise ValueError(f"unsupported audio dict shape: keys={sorted(audio_info)!r}")
    # optional fallback to non-dict case
    non_dict = cast(Mapping[str, Any], audio_info)
    try:
        sr_use = int(non_dict["sampling_rate"])
        audio = np.asarray(non_dict["array"], dtype=np.float32)
    except (TypeError, KeyError) as exc:
        raise ValueError(f"unexpected audio payload: {type(audio_info)}") from exc
    return audio, sr_use


def _to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim > 2:
        raise ValueError(f"expected 1D/2D audio, got shape {audio.shape}")

    x = np.asarray(audio, dtype=np.float32)
    if sample_rate != _TARGET_SR:
        x = librosa.resample(x, orig_sr=sample_rate, target_sr=_TARGET_SR)
    return np.asarray(x, dtype=np.float32)


def hf_audio_to_benchmark_wav(audio_info: object, dest_wav: Path) -> None:
    """Decode *audio_info*, write 16 kHz mono PCM16 WAV."""
    wave, sr = decode_audio(audio_info)
    pcm = _to_mono_16k(wave, sr)
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest_wav), pcm, _TARGET_SR, subtype="PCM_16")
