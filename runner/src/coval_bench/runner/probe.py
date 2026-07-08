# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-persisting benchmark probe.

Runs dataset samples through the real per-item pipeline (``_run_stt_item`` /
``_run_tts_item``) with ``writer=None``, so it measures the same metrics as a
production run without writing to the database. Used for private latency checks
of ``PENDING`` models.
"""

from __future__ import annotations

import asyncio
import random
import statistics
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import structlog

from coval_bench.datasets.loader import load_stt_dataset, load_tts_dataset
from coval_bench.db.models import ResultStatus
from coval_bench.registries import METRIC_SPECS, Benchmark, Metric
from coval_bench.runner.orchestrator import _run_stt_item, _run_tts_item

if TYPE_CHECKING:
    from coval_bench.config import Settings
    from coval_bench.registries import RegisteredModel

logger = structlog.get_logger("coval_bench.probe")

_STT_METRICS = (Metric.TTFS, Metric.TTFT, Metric.AUDIO_TO_FINAL, Metric.RTF, Metric.WER)
_TTS_METRICS = (Metric.TTFA, Metric.WER)
# Fixed seed so repeated probes draw the same sample for run-to-run comparability.
_SEED = 0


def _pct(values: list[float], q: float) -> float | None:
    s = sorted(values)
    if not s:
        return None
    k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[k]


def _aggregate(rows: list[Any], metrics: tuple[Metric, ...]) -> dict[str, dict[str, Any]]:
    """Summarise Result rows per metric: counts plus mean/median/p90/min/max."""
    out: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        sel = [r for r in rows if r.metric_type == metric]
        ok = [
            r.metric_value
            for r in sel
            if r.status == ResultStatus.SUCCESS and r.metric_value is not None
        ]
        stat: dict[str, Any] = {"n_ok": len(ok), "n": len(sel), "unit": METRIC_SPECS[metric].units}
        if ok:
            stat.update(
                mean=statistics.mean(ok),
                median=statistics.median(ok),
                p90=_pct(ok, 0.9),
                min=min(ok),
                max=max(ok),
            )
        out[metric.value] = stat
    return out


async def _run_model(
    *,
    entry: RegisteredModel,
    items: list[Any],
    runner: Callable[..., Awaitable[list[Any]]],
    metrics: tuple[Metric, ...],
    sem: asyncio.Semaphore,
    settings: Settings,
) -> dict[str, Any]:
    tasks = [
        runner(entry=entry, item=item, run_id=0, sem=sem, settings=settings, writer=None)
        for item in items
    ]
    rows: list[Any] = []
    for batch in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(batch, list):
            rows.extend(batch)
        else:
            logger.warning(
                "probe_item_raised",
                provider=entry.provider,
                model=entry.model,
                exc_info=batch,
            )
    return {
        "benchmark": entry.benchmark.value,
        "n_items": len(items),
        "metrics": _aggregate(rows, metrics),
    }


async def run_probe(
    *,
    settings: Settings,
    models: list[RegisteredModel],
    sample_size: int,
    concurrency: int = 1,
) -> dict[str, dict[str, Any]]:
    """Probe *models* over *sample_size* dataset items each; return aggregates.

    Concurrency defaults to 1; keep it there for a single pinned replica so the
    latency numbers are not inflated by self-contention.
    """
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict[str, Any]] = {}

    stt_models = [m for m in models if m.benchmark is Benchmark.STT]
    tts_models = [m for m in models if m.benchmark is Benchmark.TTS]

    if stt_models:
        ds = load_stt_dataset(
            settings.dataset_id,
            settings=settings,
            sample_size=sample_size,
            rng=random.Random(_SEED),  # noqa: S311
        )
        for entry in stt_models:
            results[f"{entry.provider}/{entry.model}"] = await _run_model(
                entry=entry,
                items=list(ds.items),
                runner=_run_stt_item,
                metrics=_STT_METRICS,
                sem=sem,
                settings=settings,
            )
    if tts_models:
        ds_tts = load_tts_dataset(
            "tts-v1",
            settings=settings,
            sample_size=sample_size,
            rng=random.Random(_SEED),  # noqa: S311
        )
        for entry in tts_models:
            results[f"{entry.provider}/{entry.model}"] = await _run_model(
                entry=entry,
                items=list(ds_tts.items),
                runner=_run_tts_item,
                metrics=_TTS_METRICS,
                sem=sem,
                settings=settings,
            )
    return results
