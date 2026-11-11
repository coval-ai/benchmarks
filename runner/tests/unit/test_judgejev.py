# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.metrics import compute_judgejev
from coval_bench.metrics.judgejev import find_differences
from coval_bench.registries import Metric
from coval_bench.runner.orchestrator import _item_rows

_REF = "please cancel my card before the fifth"
_HYP = "Please cancel my car before fifth, um"


class _Jev:
    def __init__(self, *probabilities: float) -> None:
        self.probabilities = probabilities
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.requests.append(body)
        return {"answers": {f"c{i}": {"noul": p} for i, p in enumerate(self.probabilities)}}


def test_adjacent_edits_merge_into_one_difference() -> None:
    words, differences = find_differences(
        "the butterfly flew off", "the but the fly flew of course"
    )
    assert words == 4
    assert [(d.reference_words, d.transcript_words, d.weight) for d in differences] == [
        ("butterfly", "but the fly", 3),
        ("off", "of course", 2),
    ]
    assert differences[0].where_in_reference == "the [butterfly] flew off"


async def test_differences_count_by_jevs_probability() -> None:
    jev = _Jev(0.9, 0.4)
    result = await compute_judgejev(_REF, _HYP, jev)
    (request,) = jev.requests
    assert [c["reference_words"] for c in request["state"]["changes"]] == ["card", "the"]
    assert request["questions"].keys() == {"c0", "c1"}
    assert result.weighted_errors == pytest.approx(1.3)
    assert (result.cutoff_errors, result.reference_words) == (1, 7)
    assert result.percentage == pytest.approx(130 / 7)


@pytest.mark.parametrize(
    ("ref", "hyp", "errors"),
    [("hello world", "Hello, world.", 0), ("hello world", "", 2), ("", "", 0)],
)
async def test_jev_is_not_asked_without_two_texts_to_compare(
    ref: str, hyp: str, errors: int
) -> None:
    jev = _Jev()
    assert (await compute_judgejev(ref, hyp, jev)).weighted_errors == errors
    assert jev.requests == []


def _rows() -> Any:
    entry = MagicMock(provider="deepgram", model="nova-3")
    return _item_rows(entry, 1, "stt")


async def test_no_key_scores_nothing() -> None:
    rows = _rows()
    settings = Settings(typesafe_api_key=None)
    await rows.score_judgejev(settings, _REF, _HYP, audio_filename="a.wav")
    assert rows.results == []


async def test_row_carries_the_counts_and_a_failed_request_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(typesafe_api_key=SecretStr("k"))

    async def down(_body: dict[str, Any], **_: Any) -> dict[str, Any]:
        raise RuntimeError("jev is down")

    rows = _rows()
    monkeypatch.setattr("coval_bench.metrics.ask_jev", down)
    await rows.score_judgejev(settings, _REF, _HYP, audio_filename="a.wav")
    assert rows.results == []

    monkeypatch.setattr("coval_bench.metrics.ask_jev", _Jev(0.9, 0.4))
    await rows.score_judgejev(settings, _REF, _HYP, audio_filename="a.wav")
    (row,) = rows.results
    assert (row.metric_type, row.metric_units) == (Metric.JUDGE_JEV, "percent")
    assert row.metric_value == pytest.approx(130 / 7)
    assert row.judgejev_weighted_errors == pytest.approx(1.3)
    assert (row.judgejev_cutoff_errors, row.judgejev_reference_words) == (1, 7)
