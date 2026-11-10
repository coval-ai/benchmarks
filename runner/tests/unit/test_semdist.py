# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from coval_bench.config import Settings
from coval_bench.metrics import compute_semdist
from coval_bench.registries import Metric
from coval_bench.runner.orchestrator import _item_rows

_VECTORS = {
    "hello world": [1.0, 0.0],
    "hello there": [0.6, 0.8],
    "goodbye": [-1.0, 0.0],
}


class _Embedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
        self.calls += 1
        return np.array([_VECTORS[text] for text in texts], dtype=np.float32)


@pytest.mark.parametrize(
    ("ref", "hyp", "distance", "embedded"),
    [
        ("hello world", "Hello, world.", 0.0, False),
        ("hello world", "hello there", 0.4, True),
        ("hello world", "goodbye", 2.0, True),
        ("hello world", "", 1.0, False),
        ("", "hello world", 1.0, False),
        ("", "", 0.0, False),
    ],
)
def test_compute_semdist(ref: str, hyp: str, distance: float, embedded: bool) -> None:
    embedder = _Embedder()
    assert compute_semdist(ref, hyp, embedder).distance == pytest.approx(distance)
    assert embedder.calls == int(embedded)


def _rows() -> Any:
    entry = MagicMock(provider="deepgram", model="nova-3")
    return _item_rows(entry, 1, "stt")


def test_no_model_dir_scores_nothing() -> None:
    rows = _rows()
    rows.score_semdist(Settings(), "hello world", "hello there", audio_filename="a.wav")
    assert rows.results == []


def test_embedder_failure_is_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    rows = _rows()
    settings = Settings(semdist_minilm_dir=tmp_path)
    rows.score_semdist(settings, "hello world", "hello there", audio_filename="a.wav")
    assert rows.results == []

    monkeypatch.setattr(
        "coval_bench.metrics.load_sentence_embedder", lambda _model_dir: _Embedder()
    )
    rows.score_semdist(settings, "hello world", "hello there", audio_filename="a.wav")
    (row,) = rows.results
    assert (row.metric_type, row.metric_units) == (Metric.SEMDIST_MINILM, "ratio")
    assert row.metric_value == pytest.approx(0.4)
