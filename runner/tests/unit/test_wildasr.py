# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the WildASR family selection (synthetic parquet, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from coval_bench.datasets.scripts.framework import _clean, balanced_sample  # noqa: E402
from coval_bench.datasets.scripts.hf_source import extract_parquet_audio  # noqa: E402
from coval_bench.datasets.scripts.wildasr import (  # noqa: E402
    WILDASR_ENV_SPECS,
    _family_pool,
    _Utterance,
)

# Clean row order (first appearance -> ordinal). Each transcript exercises a rule.
_T_AMB = "the ambiguous transcript sample"  # ordinal 0 — two DISTINCT clean recordings
_T_SHORT = "too short"  # ordinal 1 — under the word floor
_T_A = "alpha bravo charlie delta"  # ordinal 2
_T_B = "bravo charlie delta echo"  # ordinal 3
_T_C = "charlie delta echo foxtrot"  # ordinal 4
_T_D = "delta echo foxtrot golf"  # ordinal 5 — noise_gap chosen condition too long
_T_E = "echo foxtrot golf hotel"  # ordinal 6 — dup rows share a hash; extra far_field draw
_UNIQUE = (_T_AMB, _T_SHORT, _T_A, _T_B, _T_C, _T_D, _T_E)

_Row = tuple[str, float, str]  # (transcript, duration, audio_hash_id)


def _write_split(pq_dir: Path, variant: str, rows: list[_Row]) -> None:
    table = pa.table(
        {
            "transcript": [t for t, _, _ in rows],
            "duration": [d for _, d, _ in rows],
            "audio_hash_id": [h for _, _, h in rows],
        }
    )
    name = f"data__environment_degradation__en__fleurs_{variant}_en-00000-of-00001.parquet"
    pq.write_table(table, pq_dir / name)


def _block_rows(prefix: str, k: int) -> list[_Row]:
    """Block layout: the whole clean sequence repeated once per condition."""
    return [(t, 5.0, f"{prefix}{c}-{t}") for c in range(k) for t in _UNIQUE]


def _interleaved_rows(prefix: str, k: int) -> list[_Row]:
    """Interleaved layout: k consecutive condition rows per utterance."""
    return [(t, 5.0, f"{prefix}{c}-{t}") for t in _UNIQUE for c in range(k)]


@pytest.fixture
def source(tmp_path: Path) -> Path:
    pq_dir = tmp_path / "parquet"
    pq_dir.mkdir()
    clean: list[_Row] = [(t, 5.0, f"clean0-{t}") for t in _UNIQUE]
    clean.append((_T_AMB, 5.0, f"clean1-{_T_AMB}"))
    clean.append((_T_E, 5.0, f"clean0-{_T_E}"))
    _write_split(pq_dir, "clean", clean)
    _write_split(pq_dir, "clipping", [(t, 5.0, f"clip0-{t}") for t in _UNIQUE])
    _write_split(pq_dir, "far_field", _interleaved_rows("ff", 3) + [(_T_E, 5.0, f"ff3-{_T_E}")])
    _write_split(pq_dir, "reverberation", _interleaved_rows("rv", 3))
    _write_split(pq_dir, "phone_codec", _block_rows("pc", 2) + [(_T_E, 5.0, f"pc0-{_T_E}")])
    # _T_D's chosen noise_gap condition (5 % 4 = 1) pushed past the ceiling
    noise_gap = _interleaved_rows("ng", 4)
    long_row = _UNIQUE.index(_T_D) * 4 + 1
    noise_gap[long_row] = (_T_D, 16.0, noise_gap[long_row][2])
    _write_split(pq_dir, "noise_gap", noise_gap)
    return tmp_path


def test_family_pool_drops_and_survivors(source: Path) -> None:
    """Ambiguous, short, and family-wide duration violations drop; the rest survive."""
    pool = _family_pool(source)
    assert [u.transcript for u in pool] == [_T_A, _T_B, _T_C, _T_E]


def test_condition_rotation_uses_prefilter_ordinal(source: Path) -> None:
    """Chosen condition = clean-order ordinal mod that utterance's condition count."""
    pool = _family_pool(source)
    by_transcript = {u.transcript: u for u in pool}
    assert by_transcript[_T_A].chosen["phone_codec"].condition_idx == 2 % 2
    assert by_transcript[_T_B].chosen["phone_codec"].condition_idx == 3 % 2
    assert by_transcript[_T_A].chosen["noise_gap"].condition_idx == 2 % 4
    assert by_transcript[_T_C].chosen["far_field"].condition_idx == 4 % 3


def test_duplicate_rows_collapse_and_uneven_conditions_kept(source: Path) -> None:
    """Same-hash rows collapse to one condition; extra distinct draws widen the count."""
    pool = _family_pool(source)
    utterance = next(u for u in pool if u.transcript == _T_E)
    assert utterance.chosen["clean"].condition_count == 1
    assert utterance.chosen["phone_codec"].condition_count == 2
    assert utterance.chosen["far_field"].condition_count == 4
    assert utterance.chosen["far_field"].condition_idx == 6 % 4


def test_condition_rows_resolve_per_layout(source: Path) -> None:
    """Condition indices map to the right physical rows in block and interleaved files."""
    pool = _family_pool(source)
    by_transcript = {u.transcript: u for u in pool}
    chosen_b = by_transcript[_T_B].chosen["phone_codec"]
    assert chosen_b.condition_idx == 1
    assert chosen_b.row.shard_row == len(_UNIQUE) + _UNIQUE.index(_T_B)
    chosen_a = by_transcript[_T_A].chosen["noise_gap"]
    assert chosen_a.condition_idx == 2
    assert chosen_a.row.shard_row == _UNIQUE.index(_T_A) * 4 + 2
    assert chosen_a.row.audio_hash_id == f"ng2-{_T_A}"


def test_selection_pipeline_aligns_across_variants(source: Path) -> None:
    """Running each spec's parse through the framework selection yields the same
    utterances in the same order — the property that makes filenames pair up."""
    orders: list[list[str]] = []
    for spec in WILDASR_ENV_SPECS.values():
        clips = spec.parse(source)
        clips = _clean(clips, dur_min=spec.dur_min, dur_max=spec.dur_max, min_words=spec.min_words)
        selected = balanced_sample(
            clips, num=spec.num, dedup_key=spec.dedup_key, balance_dims=spec.balance_dims
        )
        orders.append([c.transcript for c in selected])
    assert len(orders) == 6
    assert all(order == orders[0] for order in orders)
    assert sorted(orders[0]) == orders[0]


def test_family_pool_is_deterministic(source: Path) -> None:
    def key(pool: list[_Utterance]) -> list[tuple[str, int, int]]:
        return [
            (u.transcript, u.chosen["noise_gap"].row.shard_row, u.chosen["noise_gap"].condition_idx)
            for u in pool
        ]

    assert key(_family_pool(source)) == key(_family_pool(source))


def test_extract_parquet_audio_writes_selected_rows(tmp_path: Path) -> None:
    """Audio bytes land at each selected clip's audio_path, keyed by _pq/_row."""
    from coval_bench.datasets.scripts.framework import Clip

    table = pa.table(
        {
            "transcript": ["one two three", "four five six"],
            "audio": [
                {"bytes": b"first-bytes", "path": "a.wav"},
                {"bytes": b"second-bytes", "path": "b.wav"},
            ],
        }
    )
    shard = tmp_path / "shard.parquet"
    pq.write_table(table, shard)
    clip = Clip(
        audio_path=tmp_path / "out.wav",
        transcript="four five six",
        meta={"_pq": str(shard), "_row": 1},
    )
    extract_parquet_audio([clip], "audio")
    assert clip.audio_path.read_bytes() == b"second-bytes"
