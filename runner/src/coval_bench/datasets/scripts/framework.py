# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Standardized dataset build/clean framework.

Every dataset is built through the same shared flow (see
``docs/standardized-dataset-cleaning.md``):

    download + extract → parse → clean → select/balance → transcode + hash →
    annotate (SileroVAD end-of-speech, a separate tool) → render manifest → upload

A dataset is a :class:`DatasetSpec` — config plus a couple of hooks (``download``,
``parse``); everything else is shared here. Adding a dataset = fill a spec + write
one ``parse``.
"""

from __future__ import annotations

import hashlib
import importlib.resources as impres
import json
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
import structlog

from coval_bench.config import DATASET_ALL

if TYPE_CHECKING:
    import google.cloud.storage as _gcs_storage

logger = structlog.get_logger(__name__)

_TARGET_SR = 16_000
_BUCKET_DEFAULT = "coval-benchmarks-datasets"
_NORM_TARGET_RMS_DBFS = -20.0
_NORM_PEAK_CEILING = 0.985  # gain cap: peaks stay below full scale
_TAG_WARN = 0.95  # per-dim coverage below this → warn + flag (still builds)
_TAG_FLOOR = 0.50  # per-dim coverage below this → abort (metadata too sparse)
# Framework owns these manifest fields; a same-named dataset meta column must not
# overwrite them (path/sha256 are assigned here, duration/vad-offset are computed).
_RESERVED_ITEM_KEYS = frozenset(
    {"path", "sha256", "transcript", "duration_sec", "speech_end_offset_ms"}
)


@dataclass
class Clip:
    """One candidate clip, shared across datasets.

    ``audio_path``/``transcript``/``meta``/``duration_sec`` come from ``parse``;
    ``filename``/``wav_path``/``sha256`` are filled during transcode. ``meta`` holds
    per-dataset provenance (speaker_id, scenario, gender, …) emitted into the manifest
    item; keys starting with ``_`` are build-internal and are NOT emitted.
    """

    audio_path: Path
    transcript: str
    meta: dict[str, object]
    duration_sec: float = 0.0
    filename: str = ""
    wav_path: Path = field(default_factory=Path)
    sha256: str = ""


@dataclass
class DatasetSpec:
    """Config + hooks for one dataset's build."""

    dataset_id: str  # "s2s-v1" → prefix "s2s-v1/audio" + manifest "s2s-v1.json"
    cache_name: str  # subdir under ~/.cache/coval-bench for this dataset's source
    download: Callable[[Path], Path]  # download + extract → source dir
    parse: Callable[[Path], list[Clip]]  # source dir → clips (duration + meta set)
    dur_min: float  # clean: keep clips within [dur_min, dur_max]
    dur_max: float
    min_words: int  # clean: minimum transcript word count
    num: int | None  # final clip count; None = every clip that survives selection
    dedup_key: Callable[[Clip], object]  # keep one clip per key
    balance_dims: Sequence[Callable[[Clip], object]]  # sample even across these (None = untagged)
    license: str
    source: str
    needs_vad_offset: bool  # run precompute_vad_offsets.py after the build?
    fetch: Callable[[list[Clip]], None] | None = None  # pull audio for selected clips only
    normalize_audio: bool = False  # loudness-normalize each clip during transcode


# --- shared build helpers ---------------------------------------------------


def _hash_file(path: Path) -> str:
    """SHA256 hex digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(clips: list[Clip], *, dur_min: float, dur_max: float, min_words: int) -> list[Clip]:
    """Clean: drop clips with no transcript, under the word floor, or outside the band."""
    return [
        c
        for c in clips
        if c.transcript.strip()
        and len(c.transcript.split()) >= min_words
        and dur_min <= c.duration_sec <= dur_max
    ]


def balanced_sample(
    clips: list[Clip],
    *,
    num: int | None,
    dedup_key: Callable[[Clip], object],
    balance_dims: Sequence[Callable[[Clip], object]],
    tag_warn: float = _TAG_WARN,
    tag_floor: float = _TAG_FLOOR,
) -> list[Clip]:
    """Select: dedup, exclude clips untagged on any balance dim, then stratified
    round-robin to *num* so the balance dimensions are evenly represented.

    ``num=None`` keeps every clip that survives dedup and the untagged exclusion,
    still in the deterministic round-robin order.

    Each dim returns a value or ``None`` (untagged/unexpected). Per-dim coverage is
    logged; below *tag_warn* it warns, below *tag_floor* it aborts. Untagged clips
    are excluded and flagged — never silently defaulted. Deterministic.
    """
    seen: set[object] = set()
    dedup: list[Clip] = []
    for clip in sorted(clips, key=lambda c: str(dedup_key(c))):
        key = dedup_key(clip)
        if key not in seen:
            seen.add(key)
            dedup.append(clip)
    if not dedup:
        raise ValueError("balanced_sample: no clips to sample from")

    # Coverage per balance dim: health check, not a gate on balancing.
    for i, dim in enumerate(balance_dims):
        coverage = sum(1 for c in dedup if dim(c) is not None) / len(dedup)
        logger.info("balance_dim_coverage", dim=i, coverage=round(coverage, 3))
        if coverage < tag_floor:
            raise ValueError(
                f"balance dim {i} coverage {coverage:.3f} below floor {tag_floor} "
                "— source metadata too sparse to balance"
            )
        if coverage < tag_warn:
            logger.warning("balance_dim_low_coverage", dim=i, coverage=round(coverage, 3))

    # Always exclude clips untagged on any balance dim; flag them.
    pool = [c for c in dedup if all(dim(c) is not None for dim in balance_dims)]
    if len(pool) < len(dedup):
        logger.warning("balance_excluded_untagged", count=len(dedup) - len(pool))

    # Stratify by the cross-product of dim values; round-robin to num.
    strata: dict[tuple[object, ...], list[Clip]] = {}
    for clip in pool:
        strata.setdefault(tuple(dim(clip) for dim in balance_dims), []).append(clip)
    for group in strata.values():
        group.sort(key=lambda c: str(dedup_key(c)))

    keys = sorted(strata, key=lambda k: tuple(str(v) for v in k))
    target = len(pool) if num is None else num
    selected: list[Clip] = []
    cursor = 0
    while len(selected) < target and any(strata[k] for k in keys):
        key = keys[cursor % len(keys)]
        if strata[key]:
            selected.append(strata[key].pop(0))
        cursor += 1
    if len(selected) < target:
        raise ValueError(f"balanced_sample: {len(selected)} clips after balancing, need {target}")
    return selected


def _loudness_gain(data: object) -> float:
    """Gain to reach the RMS target, peak-capped; silent clips pass through unchanged."""
    rms = float((data**2).mean()) ** 0.5  # type: ignore[operator]
    peak = float(abs(data).max())  # type: ignore[arg-type]
    if rms <= 0.0 or peak <= 0.0:
        return 1.0
    target_rms = float(10 ** (_NORM_TARGET_RMS_DBFS / 20.0))
    return float(min(target_rms / rms, _NORM_PEAK_CEILING / peak))


def _transcode_and_hash(clips: list[Clip], work_dir: Path, *, normalize: bool = False) -> None:
    """Transcode each clip to 16 kHz mono PCM_16 WAV; fill filename/wav_path/sha256.

    Reads float32, downmixes multi-channel to mono, and resamples off-rate sources
    to 16 kHz so any dataset's audio normalizes identically. When *normalize* is set,
    each clip is loudness-normalized (RMS target with a peak guard) before writing.
    """
    for index, clip in enumerate(clips, start=1):
        clip.filename = f"{index:04d}.wav"
        clip.wav_path = work_dir / clip.filename
        data, samplerate = sf.read(str(clip.audio_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if samplerate != _TARGET_SR:
            from scipy.signal import resample_poly

            data = resample_poly(data, _TARGET_SR, samplerate)
        if normalize:
            data = data * _loudness_gain(data)
        sf.write(str(clip.wav_path), data, _TARGET_SR, subtype="PCM_16")
        clip.sha256 = _hash_file(clip.wav_path)


def _public_meta(meta: dict[str, object]) -> dict[str, object]:
    """Meta emitted into a manifest item: drop build-internal (_) and reserved keys.

    Reserved keys are framework-owned; keeping a dataset's same-named column out of
    the spread stops it silently overwriting the computed path/sha256/duration/etc.
    """
    return {k: v for k, v in meta.items() if not k.startswith("_") and k not in _RESERVED_ITEM_KEYS}


def _render_manifest(spec: DatasetSpec, clips: list[Clip]) -> str:
    """Render the manifest JSON (shared schema + per-dataset ``meta``)."""
    manifest = {
        "_license": spec.license,
        "id": spec.dataset_id,
        "version": "1.0.0",
        "license": spec.license,
        "source": spec.source,
        "items": [
            {
                "path": f"audio/{c.filename}",
                "sha256": c.sha256,
                "transcript": c.transcript,
                "duration_sec": round(c.duration_sec, 6),
                "speech_end_offset_ms": None,
                **_public_meta(c.meta),
            }
            for c in clips
        ],
    }
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def _gcs_client() -> _gcs_storage.Client:
    import google.cloud.storage as storage

    return storage.Client()


def _upload_clips(
    clips: Sequence[Clip],
    bucket_name: str,
    *,
    prefix: str,
    overwrite: bool,
    client: _gcs_storage.Client | None = None,
) -> None:
    """Upload each WAV to gs://<bucket>/<prefix>/, then re-download + verify SHA256."""
    if client is None:
        client = _gcs_client()
    bucket = client.bucket(bucket_name)
    for clip in clips:
        blob_name = f"{prefix}/{clip.filename}"
        blob = bucket.blob(blob_name)
        if overwrite:
            blob.upload_from_filename(str(clip.wav_path), content_type="audio/wav")
        else:
            blob.upload_from_filename(
                str(clip.wav_path), content_type="audio/wav", if_generation_match=0
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            blob.download_to_filename(str(tmp_path))
            if _hash_file(tmp_path) != clip.sha256:
                blob.delete()  # remove the corrupt object so a re-run isn't blocked
                raise RuntimeError(f"upload integrity failure for {blob_name}")
        finally:
            tmp_path.unlink(missing_ok=True)


def _write_manifest(json_text: str, dataset_id: str) -> Path:
    """Write the manifest to the packaged ``<dataset_id>.json``."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_id):
        raise ValueError(f"invalid dataset_id {dataset_id!r}: use letters, digits, '.', '_', '-'")
    if dataset_id == DATASET_ALL:
        raise ValueError(f"dataset_id {DATASET_ALL!r} is reserved for pooled aggregates")
    pkg = impres.files("coval_bench.datasets.manifests")
    dest = Path(str(pkg.joinpath(f"{dataset_id}.json")))
    dest.write_text(json_text, encoding="utf-8")
    return dest


def run_build(
    spec: DatasetSpec,
    *,
    bucket: str = _BUCKET_DEFAULT,
    dry_run: bool,
    overwrite: bool,
    cache_root: Path,
) -> None:
    """Run the shared build for *spec* (see docs/standardized-dataset-cleaning.md).

    ``speech_end_offset_ms`` (SileroVAD) is left null here and filled by
    ``scripts/precompute_vad_offsets.py`` afterward when ``spec.needs_vad_offset``.
    """
    source = spec.download(cache_root)  # download + extract
    clips = spec.parse(source)  # parse → clips
    clips = _clean(clips, dur_min=spec.dur_min, dur_max=spec.dur_max, min_words=spec.min_words)
    selected = balanced_sample(  # dedup + exclude untagged + balance
        clips,
        num=spec.num,
        dedup_key=spec.dedup_key,
        balance_dims=spec.balance_dims,
    )
    if spec.fetch is not None:
        spec.fetch(selected)  # fetch audio for the selected clips only
    with tempfile.TemporaryDirectory(prefix="coval-bench-build-") as work_dir:
        _transcode_and_hash(selected, Path(work_dir), normalize=spec.normalize_audio)  # transcode
        manifest_json = _render_manifest(spec, selected)  # render manifest
        if dry_run:
            print(manifest_json, end="")  # noqa: T201 (dry-run: emit manifest for inspection)
            return
        _upload_clips(selected, bucket, prefix=f"{spec.dataset_id}/audio", overwrite=overwrite)
    _write_manifest(manifest_json, spec.dataset_id)
    # annotate: run precompute_vad_offsets.py if spec.needs_vad_offset
