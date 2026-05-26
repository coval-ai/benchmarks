# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from coval_bench.datasets.manifest import Manifest
from coval_bench.datasets.scripts.publish_reviewed_stt_v2 import (
    _build_manifest_dict,
    _hash_file,
)


def test_build_manifest_hashes_and_transcripts(tmp_path: Path) -> None:
    wav_dir = tmp_path / "wav"
    rel = Path("sub") / "clip.wav"
    dest = wav_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(16_000, dtype=np.float32)
    sf.write(str(dest), audio, 16_000, subtype="PCM_16")

    expected_sha = _hash_file(dest)

    review = {
        "sub/clip.wav": {
            "path": "sub/clip.wav",
            "index": 1,
            "approved_transcript": "ONE TWO THREE",
        },
    }

    d = _build_manifest_dict(review_by_path=review, wav_root=wav_dir)
    assert d["id"] == "stt-v2"
    assert len(d["items"]) == 1
    item = d["items"][0]
    assert item["path"] == "sub/clip.wav"
    assert item["sha256"] == expected_sha
    assert item["transcript"] == "ONE TWO THREE"
    assert item["duration_sec"] == 1.0
    assert item["gender"] == "unspecified"
    assert item["language"] == "english"

    text = json.dumps(d, indent=2)
    Manifest.model_validate_json(text)


def test_manifest_item_order_follows_review_index(tmp_path: Path) -> None:
    wav_dir = tmp_path / "wav"

    def write_wav(rel: str, index: int) -> None:
        p = wav_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        n = 1000 + index
        sf.write(str(p), np.zeros(n, dtype=np.float32), 16_000, subtype="PCM_16")

    write_wav("a.wav", 2)
    write_wav("b.wav", 0)
    write_wav("c.wav", 1)

    review = {
        "a.wav": {"path": "a.wav", "index": 2, "approved_transcript": "A"},
        "b.wav": {"path": "b.wav", "index": 0, "approved_transcript": "B"},
        "c.wav": {"path": "c.wav", "index": 1, "approved_transcript": "C"},
    }
    built = _build_manifest_dict(review_by_path=review, wav_root=wav_dir)
    paths = [item["path"] for item in built["items"]]
    assert paths == ["b.wav", "c.wav", "a.wav"]


def test_build_manifest_carries_minds14_meta(tmp_path: Path) -> None:
    """intent_class including 0, intent_name, and optional gender propagate to manifest."""
    wav_dir = tmp_path / "wav"
    dest = wav_dir / "clip.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), np.zeros(3200, dtype=np.float32), 16_000, subtype="PCM_16")

    review = {
        "clip.wav": {
            "path": "clip.wav",
            "index": 0,
            "approved_transcript": "hello",
            "intent_class": 0,
            "intent_name": "abroad",
            "gender": "F",
        },
    }
    d = _build_manifest_dict(review_by_path=review, wav_root=wav_dir)
    item = d["items"][0]
    assert item["intent_class"] == 0
    assert item["intent_name"] == "abroad"
    assert item["gender"] == "F"
    assert item["language"] == "english"
    Manifest.model_validate_json(json.dumps(d))
