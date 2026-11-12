# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality sweep: every item of every dataset through a model, once per model version.

A run opens a sweep for each model version and dataset version that has none,
then sends the items each unfinished sweep is missing to the provider, one at a
time, and stores what comes back. It stops starting new items when its time is
up, and the next run carries on from the stored results.

Each provider call holds the model lock, the same one the latency probe takes.
A clean transcript is scored for WER as soon as it's stored.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import structlog

from coval_bench.datasets.loader import dataset_manifest, stt_item_fetcher
from coval_bench.datasets.suite import SWEEP_STT_DATASETS
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.model_lock import model_lock, model_lock_key
from coval_bench.db.registry_store import RegistryStore, _registered
from coval_bench.db.sweep_store import (
    Score,
    complete_sweep,
    ensure_sweeps,
    store_result,
    store_score,
    stored_samples,
    unfinished_sweeps,
)
from coval_bench.metrics import compute_wer
from coval_bench.metrics.wer import NORM_VERSION
from coval_bench.providers._http_session import close_all as _close_http_clients
from coval_bench.registries import Benchmark, Metric
from coval_bench.runner.orchestrator import (
    _STT_TIMEOUT_S,
    _get_stt_providers,
    _read_pcm16,
    _select_models,
    _stt_silent_failure,
    _trim_to_speech_end,
    _truncate,
)

if TYPE_CHECKING:
    from coval_bench.config import Settings
    from coval_bench.datasets.manifest import STTManifestItem
    from coval_bench.registries import RegisteredModel

logger = structlog.get_logger("coval_bench.sweep")


@dataclass
class SweepSummary:
    models: int = 0
    opened: int = 0
    called: int = 0
    failed: int = 0
    scored: int = 0
    completed: int = 0


@dataclass(frozen=True)
class _Output:
    transcript: str | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _wer_score(reference: str, transcript: str) -> Score:
    wer = compute_wer(reference, transcript)
    return Score(
        metric=Metric.WER,
        metric_version=NORM_VERSION,
        errors=wer.substitutions + wer.deletions + wer.insertions,
        reference_units=wer.reference_words,
    )


async def _call(
    entry: RegisteredModel,
    pcm: bytes,
    *,
    settings: Settings,
    pool: Any,  # noqa: ANN401 — the connection pool
) -> _Output | None:
    """One provider call for one item. None when another job held the model too long."""
    provider_cls = _get_stt_providers()[entry.provider]
    key = model_lock_key(entry, provider_cls)
    async with model_lock(pool, key, timeout_s=_STT_TIMEOUT_S) as held:
        if not held:
            return None
        async with asyncio.timeout(_STT_TIMEOUT_S):
            result = await provider_cls(settings, entry.model).measure_ttft(pcm, 1, 2, 16000, 0.1)
    return _Output(
        transcript=result.complete_transcript,
        timings={
            "ttft_seconds": result.ttft_seconds,
            "audio_to_final_seconds": result.audio_to_final_seconds,
        },
        error=result.error or _stt_silent_failure(result),
    )


async def run_sweeps(
    *,
    settings: Settings,
    only: frozenset[str] | None = None,
    max_minutes: float,
) -> SweepSummary:
    """Open any missing sweeps for the collected shared STT models, then work on them."""
    summary = SweepSummary()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_minutes * 60
    async with lifespan_pool(settings) as pool:
        records = [
            r for r in await RegistryStore(pool).list_models() if r.modality is Benchmark.STT
        ]
        ids = {(r.provider, r.model): r.id for r in records}
        models, _ = _select_models(
            [_registered(r) for r in records],
            [],
            source="shared",
            benchmark_kind="stt",
            only=only,
        )
        entries = {ids[(m.provider, m.model)]: m for m in models}
        manifests = {dataset_id: dataset_manifest(dataset_id) for dataset_id in SWEEP_STT_DATASETS}
        summary.models = len(models)
        summary.opened = await ensure_sweeps(
            pool,
            [
                (model_id, dataset_id, manifest.version)
                for model_id in entries
                for dataset_id, manifest in manifests.items()
            ],
        )

        try:
            for sweep in await unfinished_sweeps(pool, list(entries)):
                manifest = manifests[sweep.dataset_id]
                if manifest.version != sweep.dataset_version:
                    continue
                entry = entries[sweep.model_id]
                fetch = stt_item_fetcher(sweep.dataset_id, settings=settings)
                stored = await stored_samples(pool, sweep.id)
                log = logger.bind(
                    provider=entry.provider,
                    model=entry.model,
                    dataset=sweep.dataset_id,
                    sweep_id=sweep.id,
                )
                left = 0
                for raw_item in cast("list[STTManifestItem]", manifest.items):
                    sample_id = raw_item.sample_id or raw_item.path
                    if sample_id in stored:
                        continue
                    if loop.time() >= deadline:
                        return summary
                    try:
                        item = await asyncio.to_thread(fetch, raw_item)
                        pcm, _ = _trim_to_speech_end(
                            _read_pcm16(item.path), item.duration_sec, item.speech_end_offset_ms
                        )
                    except Exception:
                        left += 1
                        log.exception("sweep_item_not_loaded", sample_id=sample_id)
                        continue
                    try:
                        output = await _call(entry, pcm, settings=settings, pool=pool)
                    except Exception as exc:
                        output = _Output(error=_truncate(str(exc) or type(exc).__name__))
                    if output is None:
                        left += 1
                        log.warning("sweep_item_skipped", sample_id=sample_id, reason="model busy")
                        continue
                    result_id = await store_result(
                        pool,
                        sweep_id=sweep.id,
                        sample_id=sample_id,
                        transcript=output.transcript,
                        timings=output.timings,
                        error=output.error,
                    )
                    summary.called += 1
                    if output.error is not None:
                        summary.failed += 1
                        log.warning("sweep_item_failed", sample_id=sample_id, error=output.error)
                    elif result_id is not None and output.transcript is not None:
                        try:
                            score = _wer_score(raw_item.transcript, output.transcript)
                        except Exception:
                            log.exception("sweep_item_not_scored", sample_id=sample_id)
                        else:
                            await store_score(pool, result_id, score)
                            summary.scored += 1
                if left == 0:
                    await complete_sweep(pool, sweep.id)
                    summary.completed += 1
                    log.info("sweep_completed", items=len(manifest.items))
        finally:
            await _close_http_clients()
    return summary
