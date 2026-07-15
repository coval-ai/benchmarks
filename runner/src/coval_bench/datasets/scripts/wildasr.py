# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""WildASR environment-degradation family — six index-aligned STT datasets.

The six ``environment_degradation`` splits of `bosonai/WildASR
<https://huggingface.co/datasets/bosonai/WildASR>`_ derive from the same clean
FLEURS utterances: ``clean`` holds one recording per utterance, and each degraded
split holds one or more degradation *conditions* per utterance (the source publishes
no condition column — condition identity is positional). The family builds are
index-aligned so clean and degraded rows can be compared per utterance:

- the utterance set is selected once, against the clean split, and shared by every
  family manifest — same selection, same ordering, same filenames;
- the source repeats most utterances as exact duplicate rows (same
  ``audio_hash_id``), so rows are deduped by hash per (split, transcript): a
  split's condition list is its distinct degraded audios for that utterance.
  Deterministic degradations collapse to their true condition count; randomized
  ones (noise, far-field) legitimately keep every distinct draw, so condition
  counts may vary per utterance;
- multi-condition splits contribute one row per utterance, rotated deterministically
  (utterance ordinal mod that utterance's condition count, advancing to the first
  in-band condition) so conditions are evenly represented;
- an utterance survives only if EVERY split has an in-band condition for it
  (family-wide duration filter);
- an utterance is dropped only if the clean split carries more than one DISTINCT
  recording for its transcript — transcript is the cross-split join key, and two
  clean recordings would make it ambiguous which one the degradations derive from.

Selection is deterministic from the frozen source, so each dataset can be built in
its own run and still align with its siblings.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import structlog

from coval_bench.datasets.scripts import hf_source
from coval_bench.datasets.scripts.framework import Clip, DatasetSpec

logger = structlog.get_logger(__name__)

_REPO = "bosonai/WildASR"
_CONFIG = "default"
_VARIANTS = ("clean", "clipping", "far_field", "noise_gap", "phone_codec", "reverberation")
_IDS = {
    "clean": "stt-wildasr-clean",
    "clipping": "stt-wildasr-clipping",
    "far_field": "stt-wildasr-farfield",
    "noise_gap": "stt-wildasr-noisegap",
    "phone_codec": "stt-wildasr-phonecodec",
    "reverberation": "stt-wildasr-reverb",
}
_AUDIO_COL = "audio"
_TEXT_COL = "transcript"
_DURATION_COL = "duration"
_HASH_COL = "audio_hash_id"
_DUR_MIN = 2.0
_DUR_MAX = 15.0
_MIN_WORDS = 3
_LICENSE = "Apache-2.0 (WildASR; audio derived from FLEURS, CC-BY-4.0)"


def _split(variant: str) -> str:
    return f"environment_degradation__en__fleurs_{variant}_en"


@dataclass
class _Row:
    """One source row of a split: its shard, position, and selection metadata."""

    shard: Path
    shard_row: int
    duration: float
    audio_hash_id: str


@dataclass
class _Chosen:
    """The condition row a split contributes for one utterance."""

    row: _Row
    condition_idx: int
    condition_count: int


@dataclass
class _Utterance:
    """One family utterance: the transcript plus each split's chosen row."""

    transcript: str
    chosen: dict[str, _Chosen]


def _read_split_rows(source: Path, variant: str) -> dict[str, list[_Row]]:
    """Rows of *variant* grouped by transcript, deduped by audio hash, in order.

    A transcript's row list is its distinct condition audios: consecutive for
    interleaved splits, file-order separated for block-layout splits — order of
    first appearance gives stable condition indices either way. Exact duplicate
    rows (same ``audio_hash_id``) collapse to their first occurrence; a row
    without a hash is kept as its own entry.
    """
    import pyarrow.parquet as pq

    shards = sorted((source / "parquet").glob(f"*{_split(variant)}-*.parquet"))
    if not shards:
        raise hf_source.HFUnsupported(f"{_REPO}: no cached parquet for split {_split(variant)}")
    grouped: dict[str, list[_Row]] = {}
    seen: dict[str, set[str]] = {}
    for shard in shards:
        columns = [_TEXT_COL, _DURATION_COL, _HASH_COL]
        for i, row in enumerate(pq.read_table(shard, columns=columns).to_pylist()):
            transcript = str(row[_TEXT_COL] or "").strip()
            audio_hash = str(row[_HASH_COL] or "")
            if audio_hash:
                if audio_hash in seen.setdefault(transcript, set()):
                    continue
                seen[transcript].add(audio_hash)
            grouped.setdefault(transcript, []).append(
                _Row(
                    shard=shard,
                    shard_row=i,
                    duration=hf_source.as_duration(row[_DURATION_COL]),
                    audio_hash_id=audio_hash,
                )
            )
    return grouped


def _family_pool(source: Path) -> list[_Utterance]:
    """Compute the shared utterance pool from all six splits (deterministic).

    Utterance identity is the clean transcript; ordinals (used for condition
    rotation) are assigned over clean's transcripts in row order, before any
    filtering, so a drop never shifts a survivor's chosen condition. A split's
    condition count for an utterance is its distinct-audio row count, so counts
    may vary per utterance (randomized degradations keep every distinct draw);
    the rotation advances to the first in-band condition rather than dropping
    an utterance whose rotation-point condition is too long.
    """
    split_rows = {variant: _read_split_rows(source, variant) for variant in _VARIANTS}
    shape = {
        variant: dict(Counter(len(rows) for rows in split_rows[variant].values()))
        for variant in _VARIANTS
    }
    logger.info("wildasr_split_shape", **shape)

    pool: list[_Utterance] = []
    dropped = Counter[str]()
    for ordinal, (transcript, clean_rows) in enumerate(split_rows["clean"].items()):
        if not transcript or len(transcript.split()) < _MIN_WORDS:
            dropped["words"] += 1
            continue
        if len(clean_rows) != 1:
            dropped["ambiguous"] += 1
            continue
        chosen: dict[str, _Chosen] = {}
        reason = None
        for variant in _VARIANTS:
            rows = split_rows[variant].get(transcript)
            if not rows:
                reason = "missing"
                break
            idx = next(
                (
                    (ordinal + offset) % len(rows)
                    for offset in range(len(rows))
                    if _DUR_MIN <= rows[(ordinal + offset) % len(rows)].duration <= _DUR_MAX
                ),
                None,
            )
            if idx is None:
                reason = "duration"
                break
            chosen[variant] = _Chosen(row=rows[idx], condition_idx=idx, condition_count=len(rows))
        if reason is not None:
            dropped[reason] += 1
            continue
        pool.append(_Utterance(transcript=transcript, chosen=chosen))

    logger.info("wildasr_family_pool", kept=len(pool), dropped=dict(dropped))
    if not pool:
        raise hf_source.HFUnsupported(f"{_REPO}: family selection produced no utterances")
    return pool


def _download(cache_root: Path) -> Path:
    """Download the six environment-split parquet shards into the shared cache."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise hf_source.HFUnsupported(
            "WildASR family build needs pyarrow (run with --extra hf-parquet)"
        ) from exc
    pq_dir = cache_root / "parquet"
    pq_dir.mkdir(parents=True, exist_ok=True)
    files = hf_source.list_parquet_files(_REPO, _CONFIG)
    for variant in _VARIANTS:
        split = _split(variant)
        shards = [f for f in files if Path(f).name.startswith(f"{split}-")]
        if not shards:
            raise hf_source.HFUnsupported(f"{_REPO}: no parquet shards for split {split}")
        for rel in shards:
            dest = pq_dir / rel.replace("/", "__")
            if not dest.exists():
                logger.info("wildasr_parquet_download", file=rel)
                hf_source.download_parquet(_REPO, rel, dest)
    return cache_root


def _make_parse(variant: str) -> Callable[[Path], list[Clip]]:
    def parse(source: Path) -> list[Clip]:
        audio_dir = source / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        clips: list[Clip] = []
        for utterance in _family_pool(source):
            chosen = utterance.chosen[variant]
            clips.append(
                Clip(
                    audio_path=audio_dir / f"{chosen.row.shard.stem}-{chosen.row.shard_row}.wav",
                    transcript=utterance.transcript,
                    duration_sec=chosen.row.duration,
                    meta={
                        "audio_hash_id": chosen.row.audio_hash_id,
                        "condition_idx": chosen.condition_idx,
                        "condition_count": chosen.condition_count,
                        "_pq": str(chosen.row.shard),
                        "_row": chosen.row.shard_row,
                    },
                )
            )
        return clips

    return parse


def _fetch(selected: list[Clip]) -> None:
    hf_source.extract_parquet_audio(selected, _AUDIO_COL)


def _make_spec(variant: str) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=_IDS[variant],
        cache_name="wildasr-env",
        download=_download,
        parse=_make_parse(variant),
        dur_min=_DUR_MIN,
        dur_max=_DUR_MAX,
        min_words=_MIN_WORDS,
        num=None,
        dedup_key=lambda clip: clip.transcript,
        balance_dims=(),
        license=_LICENSE,
        source=f"{_REPO} {_split(variant)}",
        needs_vad_offset=True,
        fetch=_fetch,
        normalize_audio=True,
    )


WILDASR_ENV_SPECS: dict[str, DatasetSpec] = {
    spec.dataset_id: spec for spec in (_make_spec(variant) for variant in _VARIANTS)
}
