# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Latency probe: the same short input to every collected model, once per tick.

Unlike a benchmark run this scores nothing. It makes one provider call per
model, with no retry, and stores the timings (or the failure) in
``probe_results``. A failed call is a data point, not something to hide.

Models are probed one at a time, each under its model lock, so nothing on our
side competes with the call being timed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog

from coval_bench.arena.provider_health import classify_failure
from coval_bench.datasets.loader import load_stt_dataset
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.model_lock import model_lock, model_lock_key
from coval_bench.db.probe_store import ProbeResult, insert_probe_result
from coval_bench.db.registry_store import fetch_models_with_ids
from coval_bench.metrics import compute_ttfs
from coval_bench.providers._http_session import close_all as _close_http_clients
from coval_bench.registries import METRIC_EXCLUSIONS, Benchmark, Metric
from coval_bench.runner.orchestrator import (
    _STT_TIMEOUT_S,
    _TTS_TIMEOUT_S,
    _current_tick,
    _get_stt_providers,
    _get_tts_providers,
    _read_pcm16,
    _select_models,
    _stt_silent_failure,
    _trim_to_speech_end,
    _truncate,
    _ttfa_transport_gate,
)
from coval_bench.runner.probe import _warmup

if TYPE_CHECKING:
    from coval_bench.config import Settings
    from coval_bench.registries import RegisteredModel

logger = structlog.get_logger("coval_bench.latency_probe")

STT_PROBE_DATASET = "stt-probe-v1"
TTS_PROBE_TEXT = "That's one small step for man, one giant leap for mankind."

Origin = Literal["provider", "runner"]


@dataclass(frozen=True)
class ProbeSummary:
    scheduled_at: datetime
    probed: int
    stored: int
    failed: int
    skipped: int
    errored: int


def _origin(error: str | None, status_code: int | None = None) -> Origin:
    """A bad key, spent credits or a rate limit is our problem, not the provider's."""
    return "runner" if classify_failure(status_code, error) is not None else "provider"


async def _call(
    build: Callable[[], Any],
    request: Callable[[Any], Awaitable[Any]],
    timeout_s: float,
) -> tuple[Any | None, str | None, Origin | None]:
    """One provider call with no retry: (result, error, origin)."""
    try:
        provider = build()
    except Exception as exc:
        return None, _truncate(str(exc) or type(exc).__name__), "runner"
    deadline = asyncio.timeout(timeout_s)
    try:
        async with deadline:
            result = await request(provider)
    except Exception as exc:
        if deadline.expired():
            return None, f"timed out after {timeout_s}s", "provider"
        error = _truncate(str(exc) or type(exc).__name__)
        return None, error, _origin(error)
    return result, None, None


def _excluded(entry: RegisteredModel, metric: Metric) -> bool:
    return (entry.provider, entry.model) in METRIC_EXCLUSIONS[metric]


def _to_ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000.0


async def _probe_stt(
    entry: RegisteredModel,
    item: Any,  # noqa: ANN401 — DatasetItem
    *,
    model_id: int,
    settings: Settings,
    scheduled_at: datetime,
) -> ProbeResult:
    provider_cls = _get_stt_providers().get(entry.provider)
    pcm, _ = _trim_to_speech_end(
        _read_pcm16(item.path), item.duration_sec, item.speech_end_offset_ms
    )

    def row(sent_at: datetime, **fields: Any) -> ProbeResult:  # noqa: ANN401
        return ProbeResult(
            model_id=model_id,
            scheduled_at=scheduled_at,
            sent_at=sent_at,
            **fields,
        )

    if provider_cls is None:
        return row(datetime.now(UTC), error="unknown STT provider", failure_origin="runner")

    sent_at = datetime.now(UTC)
    result, error, origin = await _call(
        lambda: provider_cls(settings, entry.model),
        lambda provider: provider.measure_ttft(pcm, 1, 2, 16000, 0.1),
        _STT_TIMEOUT_S,
    )
    if result is not None:
        error = result.error or _stt_silent_failure(result)
        origin = _origin(error) if error else None
    if result is None or error is not None:
        return row(sent_at, error=error or "no result", failure_origin=origin or "provider")

    measured: dict[str, float | None] = {}
    if not _excluded(entry, Metric.TTFT):
        measured["ttft_ms"] = _to_ms(result.ttft_seconds)
    offset_ms = item.speech_end_offset_ms
    if not _excluded(entry, Metric.TTFS) and offset_ms is not None:
        final = result.audio_to_final_seconds
        measured["ttfs_ms"] = (
            None if final is None else compute_ttfs(final, offset_ms / 1000.0) * 1000.0
        )
    missing = [
        name.removesuffix("_ms").upper() for name, value in measured.items() if value is None
    ]
    if missing:
        return row(
            sent_at,
            **measured,
            error=f"no {' or '.join(missing)} produced",
            failure_origin="provider",
        )
    return row(sent_at, **measured)


async def _probe_tts(
    entry: RegisteredModel,
    *,
    model_id: int,
    settings: Settings,
    scheduled_at: datetime,
) -> ProbeResult:
    provider_cls = _get_tts_providers().get(entry.provider)
    voice = entry.voices[0].id if entry.voices else entry.voice

    def row(sent_at: datetime, **fields: Any) -> ProbeResult:  # noqa: ANN401
        return ProbeResult(
            model_id=model_id,
            voice=voice,
            scheduled_at=scheduled_at,
            sent_at=sent_at,
            **fields,
        )

    if provider_cls is None:
        return row(datetime.now(UTC), error="unknown TTS provider", failure_origin="runner")

    sent_at = datetime.now(UTC)
    result, error, origin = await _call(
        lambda: provider_cls(settings=settings, model=entry.model, voice=voice),
        lambda provider: provider.synthesize(TTS_PROBE_TEXT),
        _TTS_TIMEOUT_S,
    )
    if result is not None:
        if result.audio_path is not None:
            result.audio_path.unlink(missing_ok=True)
        if result.error:
            error, origin = result.error, _origin(result.error, result.status_code)
        elif result.ttfa_ms is None:
            error, origin = "no TTFA produced", "provider"
        else:
            error = _ttfa_transport_gate(result)
            origin = "runner" if error else None
    if result is None or error is not None:
        return row(
            sent_at,
            error=error or "no result",
            failure_origin=origin or "provider",
            status_code=result.status_code if result is not None else None,
        )
    return row(sent_at, ttfa_ms=result.ttfa_ms)


async def run_latency_probes(
    *,
    settings: Settings,
    kind: Literal["stt", "tts"],
    only: frozenset[str] | None = None,
) -> ProbeSummary:
    """Probe every collected shared *kind* model once and store each outcome."""
    scheduled_at = _current_tick(settings)
    stored = failed = skipped = errored = 0
    async with lifespan_pool(settings) as pool:
        benchmark = Benchmark.STT if kind == "stt" else Benchmark.TTS
        rows = [(i, m) for i, m in await fetch_models_with_ids(pool) if m.benchmark is benchmark]
        ids = {(m.provider, m.model): i for i, m in rows}
        roster = [m for _, m in rows]
        stt_models, tts_models = _select_models(
            [m for m in roster if m.benchmark is Benchmark.STT],
            [m for m in roster if m.benchmark is Benchmark.TTS],
            source="shared",
            benchmark_kind=kind,
            only=only,
        )
        models = stt_models if kind == "stt" else tts_models
        item = (
            load_stt_dataset(STT_PROBE_DATASET, settings=settings, sample_size=1).items[0]
            if kind == "stt" and models
            else None
        )

        providers = _get_stt_providers() if kind == "stt" else _get_tts_providers()
        lock_wait_s = _STT_TIMEOUT_S if kind == "stt" else _TTS_TIMEOUT_S

        async def probe(entry: RegisteredModel) -> ProbeResult:
            model_id = ids[(entry.provider, entry.model)]
            if kind == "stt":
                return await _probe_stt(
                    entry, item, model_id=model_id, settings=settings, scheduled_at=scheduled_at
                )
            return await _probe_tts(
                entry, model_id=model_id, settings=settings, scheduled_at=scheduled_at
            )

        try:
            await _warmup(models, settings)
            for entry in models:
                log = logger.bind(
                    benchmark=kind.upper(), provider=entry.provider, model=entry.model
                )
                key = model_lock_key(entry, providers.get(entry.provider))
                try:
                    async with model_lock(pool, key, timeout_s=lock_wait_s) as held:
                        result = await probe(entry) if held else None
                    if result is None:
                        skipped += 1
                        log.warning("latency_probe_skipped", reason="model is busy")
                        continue
                    was_stored = await insert_probe_result(pool, result)
                except Exception:
                    errored += 1
                    log.exception("latency_probe_errored")
                    continue
                if not was_stored:
                    skipped += 1
                    log.warning("latency_probe_skipped", reason="tick already has a row")
                    continue
                stored += 1
                failed += result.error is not None
                log.info(
                    "latency_probe",
                    status=result.status,
                    ttft_ms=result.ttft_ms,
                    ttfs_ms=result.ttfs_ms,
                    ttfa_ms=result.ttfa_ms,
                    error=result.error,
                    failure_origin=result.failure_origin,
                )
        finally:
            await _close_http_clients()
    return ProbeSummary(
        scheduled_at=scheduled_at,
        probed=len(models),
        stored=stored,
        failed=failed,
        skipped=skipped,
        errored=errored,
    )
