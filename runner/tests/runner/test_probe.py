# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the non-persisting probe (coval_bench.runner.probe).

Dataset loading and the per-item runners are monkeypatched, so no network or DB
is touched — the focus is the aggregation and that ``writer=None`` is propagated.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from coval_bench.db.models import ResultStatus
from coval_bench.registries import Benchmark, Metric
from coval_bench.runner import probe as probe_mod


def _row(metric: Metric, value: float | None, status: ResultStatus = ResultStatus.SUCCESS) -> Any:
    return SimpleNamespace(metric_type=metric, metric_value=value, status=status)


def test_pct_and_aggregate() -> None:
    rows = [_row(Metric.TTFS, v) for v in (0.2, 0.4, 0.6, 0.8, 1.0)]
    rows.append(_row(Metric.TTFS, None, ResultStatus.FAILED))  # excluded from stats
    agg = probe_mod._aggregate(rows, (Metric.TTFS,))
    s = agg["TTFS"]
    assert s["n"] == 6
    assert s["n_ok"] == 5
    assert s["unit"] == "seconds"
    assert s["mean"] == pytest.approx(0.6)
    assert s["median"] == pytest.approx(0.6)
    assert s["min"] == 0.2
    assert s["max"] == 1.0


def test_aggregate_no_successes_omits_stats() -> None:
    rows = [_row(Metric.TTFA, None, ResultStatus.FAILED)]
    agg = probe_mod._aggregate(rows, (Metric.TTFA,))
    assert agg["TTFA"] == {"n_ok": 0, "n": 1, "unit": "milliseconds"}


@pytest.mark.asyncio
async def test_run_probe_no_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_probe routes each model through the right runner with writer=None."""
    captured: dict[str, Any] = {}

    fake_ds = SimpleNamespace(items=[object(), object(), object()])
    monkeypatch.setattr(probe_mod, "load_stt_dataset", lambda *a, **k: fake_ds)
    monkeypatch.setattr(probe_mod, "load_tts_dataset", lambda *a, **k: fake_ds)

    async def fake_stt(*, writer: Any, **_: Any) -> list[Any]:
        captured["stt_writer"] = writer
        return [_row(Metric.TTFS, 0.5), _row(Metric.WER, 0.0)]

    async def fake_tts(*, writer: Any, **_: Any) -> list[Any]:
        captured["tts_writer"] = writer
        return [_row(Metric.TTFA, 300.0)]

    monkeypatch.setattr(probe_mod, "_run_stt_item", fake_stt)
    monkeypatch.setattr(probe_mod, "_run_tts_item", fake_tts)

    models = [
        SimpleNamespace(benchmark=Benchmark.STT, provider="baseten", model="whisper-large-v3"),
        SimpleNamespace(benchmark=Benchmark.TTS, provider="baseten", model="qwen3-tts-1.7b"),
    ]
    results = await probe_mod.run_probe(
        settings=SimpleNamespace(), models=models, sample_size=3, concurrency=1
    )

    assert captured["stt_writer"] is None
    assert captured["tts_writer"] is None
    stt = results["baseten/whisper-large-v3"]
    assert stt["benchmark"] == "STT"
    assert stt["n_items"] == 3
    assert stt["metrics"]["TTFS"]["n_ok"] == 3
    assert stt["metrics"]["TTFS"]["median"] == pytest.approx(0.5)
    tts = results["baseten/qwen3-tts-1.7b"]
    assert tts["metrics"]["TTFA"]["mean"] == pytest.approx(300.0)
