# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Export reviewed MInDS-14 en-US rows to local 16 kHz mono PCM_16 WAV files.

Loads ``PolyAI/minds14`` (config ``en-US``, split ``train``), scans only the
``path`` strings listed in a review JSON (**same shape as your reviewed
artifacts**: top-level keys or ``value["path"]``), writes WAVs under
``--out-dir`` preserving HF ``path``, then prints how to SHA256/hash and upload.

Usage::

    uv run --project runner \\
        --with datasets \\
        --with torch \\
        --with torchcodec \\
        python -m coval_bench.datasets.scripts.export_minds14_reviewed_wavs \\
        export \\
        --review-json review_results.json \\
        --out-dir build/stt-v2

If decoding fails asking for torchcodec/torch, add ``--with torchcodec`` /
``--with torch`` as the error suggests.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import click
import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_TARGET_SR = 16_000
_HF_DATASET = "PolyAI/minds14"
_HF_CONFIG = "en-US"
_HF_SPLIT = "train"


def _review_index(row: dict[str, Any]) -> int:
    v = row.get("index", 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _paths_from_review_file(path: Path) -> dict[str, Any]:
    """Return map path_str -> raw review dict (unused here; kept for future)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise click.ClickException(f"Review JSON must be an object: {path}")

    by_path: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise click.ClickException(f"Review entry {key!r} must be an object")
        p = str(value.get("path", key)).strip()
        if not p:
            raise click.ClickException(f"Empty path in entry {key!r}")
        if p in by_path:
            raise click.ClickException(f"Duplicate reviewed path: {p!r}")
        by_path[p] = value
    return by_path


def _load_hf_train() -> object:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise click.ClickException(
            "Install datasets: uv run --project runner --with datasets python ..."
        ) from exc
    return load_dataset(_HF_DATASET, name=_HF_CONFIG, split=_HF_SPLIT)


def _decode_audio(audio_info: object) -> tuple[np.ndarray, int]:
    """HF ``Audio`` decoding -> (mono-ready float column, sr).

    Supports dict rows and HF ``torchcodec`` payloads that implement
    ``__getitem__(\"array\")`` / ``__getitem__(\"sampling_rate\")`` but are not
    ``dict`` instances (see ``datasets.features._torchcodec.AudioDecoder``).
    """
    if isinstance(audio_info, dict):
        if "array" in audio_info and "sampling_rate" in audio_info:
            audio = np.asarray(audio_info["array"], dtype=np.float32)
            sr = int(audio_info["sampling_rate"])
            return audio, sr

        disk_path = audio_info.get("path")
        if disk_path:
            audio, sr = sf.read(str(disk_path), dtype="float32", always_2d=False)
            return np.asarray(audio, dtype=np.float32), int(sr)

    non_dict = cast(Mapping[str, Any], audio_info)
    try:
        sr = int(non_dict["sampling_rate"])
        audio = np.asarray(non_dict["array"], dtype=np.float32)
    except (TypeError, KeyError) as exc:
        raise ValueError(f"unexpected audio payload type: {type(audio_info)}") from exc
    return audio, sr


def _to_16k_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim > 2:
        raise ValueError(f"expected 1D or 2D audio, got shape {audio.shape}")
    if sample_rate != _TARGET_SR:
        audio = librosa.resample(
            np.asarray(audio, dtype=np.float32),
            orig_sr=sample_rate,
            target_sr=_TARGET_SR,
        )
    return np.asarray(audio, dtype=np.float32)


@click.group()
def cli() -> None:
    """MInDS-14 reviewed WAV export (local files only — no upload)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def export_reviewed_wavs(review_json: Path, out_dir: Path) -> tuple[dict[str, Any], int]:
    """Export reviewed paths to 16 kHz mono PCM16 WAV under *out_dir*.

    Returns ``(review_by_path, wav_count)`` for downstream manifest hashing.
    """
    review_by_path = _paths_from_review_file(review_json)
    wanted: set[str] = set(review_by_path.keys())
    dataset = cast(Iterable[Mapping[str, Any]], _load_hf_train())

    found: dict[str, tuple[np.ndarray, int]] = {}
    for row in dataset:
        p = str(row.get("path", "")).strip()
        if p not in wanted:
            continue
        audio_info = row.get("audio")
        try:
            audio, sr = _decode_audio(audio_info)
        except Exception as exc:
            raise click.ClickException(
                f"Decode failed for path={p!r}: {exc}\n"
                "Try installing torch + torchcodec, e.g.:\n"
                "  uv run --project runner --with datasets --with torch "
                "--with torchcodec python -m ..."
            ) from exc
        found[p] = (audio, sr)

    missing = wanted - found.keys()
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        more = "" if len(missing) <= 5 else f" … (+{len(missing) - 5})"
        raise click.ClickException(
            f"{len(missing)} reviewed path(s) not found in MInDS-14 {_HF_SPLIT}: {sample}{more}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    for path_str, (audio_raw, sr) in sorted(
        found.items(),
        key=lambda item: _review_index(review_by_path[item[0]]),
    ):
        audio_16k = _to_16k_mono(audio_raw, sr)
        dest = out_dir / path_str
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), audio_16k, _TARGET_SR, subtype="PCM_16")
        exported += 1
        logger.info("Wrote [%s] %s", exported, dest)

    return review_by_path, exported


@cli.command("export")
@click.option(
    "--review-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Review JSON (object keyed by path; entries have path / approved transcript).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where to write WAVs (mirror HF path under this directory).",
)
def export_cmd(review_json: Path, out_dir: Path) -> None:
    """HF train split → WAV only for reviewed paths."""
    _, exported = export_reviewed_wavs(review_json, out_dir)

    click.echo("")
    click.echo(f"Exported {exported} WAVs under {out_dir}")
    click.echo("Next:")
    click.echo("  Run publish (manifest + hashes + optional GCS upload):")
    click.echo(
        "    uv run --project runner --with datasets --with torch --with torchcodec python -m "
        "coval_bench.datasets.scripts.publish_reviewed_stt_v2 publish "
        f"--review-json {review_json} --wav-dir {out_dir}"
    )


if __name__ == "__main__":
    cli()
