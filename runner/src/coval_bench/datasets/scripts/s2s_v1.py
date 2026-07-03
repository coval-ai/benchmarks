# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""s2s-v1 dataset adapter — builds the SLURP query-eliciting clip set via the framework.

Provides SLURP's ``download``/``parse`` plus balance config for the :class:`DatasetSpec`;
the shared build flow lives in ``framework.py``. Cleaning (duration band + min words)
is handled generically by the framework, so there is no backfill here — selection
picks only from already-clean clips.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import soundfile as sf

from coval_bench.datasets.scripts.framework import Clip, DatasetSpec

_META_BASE = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp"
_AUDIO_URL = "https://zenodo.org/record/4274930/files/slurp_real.tar.gz"
_SPLITS = ("train", "devel", "test")  # real splits only — excludes train_synthetic
_NUM = 50
_CHUNK = 1024 * 1024
_QUERY_RE = re.compile(  # query intents only — clips must elicit a spoken reply
    r"query|qa_|factoid|definition|recommend|currency|stock|maths|convert|traffic|joke|quirky"
)


def download_slurp(cache_root: Path) -> Path:
    """Download SLURP metadata (GitHub) + audio (Zenodo) into *cache_root*; return it."""
    meta = cache_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for name in (*(f"{s}.jsonl" for s in _SPLITS), "metadata.json"):
        dest = meta / name
        if dest.exists():
            continue
        req = urllib.request.Request(  # noqa: S310 (audited: hardcoded https SLURP URL)
            f"{_META_BASE}/{name}", headers={"User-Agent": "coval-bench/build-dataset"}
        )
        # timeout = idle/stall limit (not a total deadline); a slow large transfer is fine.
        with urllib.request.urlopen(req, timeout=120) as response:  # noqa: S310
            dest.write_bytes(response.read())

    audio_dir = cache_root / "slurp_real"
    if audio_dir.exists():
        return cache_root
    tar_path = cache_root / "slurp_real.tar.gz"
    if not tar_path.exists():
        req = urllib.request.Request(  # noqa: S310 (audited: hardcoded https Zenodo URL)
            _AUDIO_URL, headers={"User-Agent": "coval-bench/build-dataset"}
        )
        # Download to a .part file and rename on success so an interrupted run never
        # leaves a truncated archive that the tar_path.exists() check would trust.
        part = tar_path.with_suffix(".tar.gz.part")
        try:
            with (
                urllib.request.urlopen(req, timeout=120) as response,  # noqa: S310
                part.open("wb") as out,
            ):
                while chunk := response.read(_CHUNK):
                    out.write(chunk)
        except OSError:
            part.unlink(missing_ok=True)
            raise
        part.replace(tar_path)
    # Extract into a staging dir and rename on success, so a failed extraction never
    # leaves a partial slurp_real/ that the audio_dir.exists() check above would trust.
    staging = cache_root / "slurp_real.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(staging, filter="data")  # noqa: S202 (filter blocks path traversal)
    extracted = staging / "slurp_real"
    (extracted if extracted.exists() else staging).replace(audio_dir)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    return cache_root


def _usrid_map(meta_dir: Path) -> dict[str, str]:
    """Map ``{audio_filename: usrid}`` from SLURP ``metadata.json``."""
    raw: Any = json.loads((meta_dir / "metadata.json").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in raw.values():
        for filename, rec in (entry.get("recordings") or {}).items():
            usrid = rec.get("usrid")
            if isinstance(usrid, str):
                out[str(filename)] = usrid
    return out


def parse_slurp(source: Path) -> list[Clip]:
    """Parse SLURP jsonl → Clips. Applies the mandatory per-dataset filters.

    Mandatory (drop if failing): headset recording, ``status=correct``, non-empty
    transcript, and a query/question-eliciting intent (V2V needs a spoken reply).
    gender/native are recorded as-is — untagged → ``"UNK"``/``None`` — and left for
    the framework's ``balanced_sample`` to exclude/balance; scenario is metadata only.
    """
    meta_dir = source / "metadata"
    audio_dir = source / "slurp_real"
    usrid_map = _usrid_map(meta_dir)
    clips: list[Clip] = []
    for split in _SPLITS:
        for line in (meta_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record: Any = json.loads(line)
            sentence = str(record.get("sentence") or "").strip()
            intent = str(record.get("intent") or "")
            if not sentence or not _QUERY_RE.search(intent):
                continue
            scenario = str(record.get("scenario") or "").strip()
            for rec in record.get("recordings") or []:
                filename = str(rec.get("file") or "")
                if "-headset" not in filename or rec.get("status") != "correct":
                    continue
                usrid = usrid_map.get(filename, "")
                gender = usrid[0] if usrid[:1] in ("F", "M") else "UNK"
                second = usrid[1:2]
                native = True if second == "E" else False if second == "O" else None
                flac = audio_dir / filename
                duration = float(sf.info(str(flac)).duration) if flac.exists() else 0.0
                clips.append(
                    Clip(
                        audio_path=flac,
                        transcript=sentence,
                        duration_sec=duration,
                        meta={
                            "scenario": scenario,
                            "intent": intent,
                            "speaker_id": usrid,
                            "gender": gender,
                            "native": native,
                            "slurp_id": int(record["slurp_id"]),
                            "slurp_file": filename,
                        },
                    )
                )
    return clips


def _slurp_id(clip: Clip) -> object:
    """Dedup key — one clip per SLURP prompt."""
    return clip.meta["slurp_id"]


def _gender(clip: Clip) -> object:
    """Balance value for gender; None if untagged."""
    g = clip.meta["gender"]
    return g if g in ("F", "M") else None


def _native(clip: Clip) -> object:
    """Balance value for native (True/False); None if untagged."""
    return clip.meta["native"]


S2S_V1 = DatasetSpec(
    dataset_id="s2s-v1",
    cache_name="slurp",
    download=download_slurp,
    parse=parse_slurp,
    dur_min=2.0,
    dur_max=10.0,
    min_words=3,
    num=_NUM,
    dedup_key=_slurp_id,
    balance_dims=(_gender, _native),  # scenario is metadata only, not balanced
    needs_vad_offset=True,
    license="CC-BY-NC-4.0",
    source=(
        "SLURP (Bastianelli et al., EMNLP 2020); audio via Zenodo 4274930; "
        "headset, status=correct, query-eliciting intents"
    ),
)
