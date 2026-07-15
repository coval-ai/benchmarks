# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hf_source + build helpers (pure: no network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from coval_bench.datasets.scripts.build import _hf_spec, _meta_dim
from coval_bench.datasets.scripts.framework import Clip
from coval_bench.datasets.scripts.hf_source import _config_parquet_files, as_duration


def _clip(meta: dict[str, object]) -> Clip:
    return Clip(audio_path=Path("/x.wav"), transcript="hi there now", meta=meta)


def test_as_duration_coerces_null_and_junk() -> None:
    """Null / blank / non-numeric duration cells become 0.0 instead of crashing."""
    assert as_duration(3.5) == 3.5
    assert as_duration("2.0") == 2.0
    assert as_duration(None) == 0.0
    assert as_duration("") == 0.0
    assert as_duration("n/a") == 0.0


def test_config_parquet_files_matches_config_dirs() -> None:
    files = ["data/en/train.parquet", "data/fr/train.parquet", "README.md"]
    assert _config_parquet_files(files, "en") == ["data/en/train.parquet"]


def test_config_parquet_files_single_config_data_layout() -> None:
    """Single-config repos publish data/<split>-*.parquet under the 'default' config."""
    files = ["data/train-00000-of-00001.parquet", "README.md", ".gitattributes"]
    assert _config_parquet_files(files, "default") == ["data/train-00000-of-00001.parquet"]
    assert _config_parquet_files(files, "en") == []


def _download(root: Path) -> Path:
    return root


def _parse(_source: Path) -> list[Clip]:
    return []


def test_hf_spec_dedups_by_transcript() -> None:
    """Same recording published under two row indices collapses to one clip."""
    hooks = (_download, _parse, None)
    with patch("coval_bench.datasets.scripts.build._resolve_hooks", return_value=hooks):
        spec = _hf_spec(
            "org/ds",
            config=None,
            split=None,
            audio_col=None,
            text_col=None,
            balance_cols=(),
            num=50,
            dur_min=2.0,
            dur_max=10.0,
            dataset_id=None,
            license_id=None,
            source_label=None,
            normalize=False,
        )
    first = Clip(audio_path=Path("/split-0.wav"), transcript="same words here", meta={})
    dupe = Clip(audio_path=Path("/split-9.wav"), transcript="same words here", meta={})
    assert spec.dedup_key(first) == spec.dedup_key(dupe)


def test_meta_dim_keeps_false_and_zero() -> None:
    """Balancing on a boolean/numeric column keeps False/0 (only missing → untagged)."""
    native = _meta_dim("native")
    assert native(_clip({"native": False})) is False
    assert native(_clip({"native": True})) is True
    assert _meta_dim("count")(_clip({"count": 0})) == 0
    assert native(_clip({})) is None
    assert native(_clip({"native": ""})) is None
