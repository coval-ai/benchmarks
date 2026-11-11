# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Latency probe: the same short input to every collected model, once per tick.

Unlike a benchmark run this scores nothing. It makes one provider call per
model, with no retry, and stores the timings (or the failure) in
``probe_results``. A failed call is a data point, not something to hide.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog

from coval_bench.datasets.loader import load_stt_dataset
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.probe_store import ProbeResult, insert_probe_result
from coval_bench.db.registry_store import fetch_models
from coval_bench.metrics import compute_ttfs
from coval_bench.providers._http_session import close_all as _close_http_clients
from coval_bench.registries import METRIC_EXCLUSIONS, Benchmark, Metric
from coval_bench.runner.gate import ModelGate
from coval_bench.runner.orchestrator import (
    _CONCURRENCY_CAP,
    _STT_TIMEOUT_S,
    _TTS_TIMEOUT_S,
    _current_tick,
    _gate_key,
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
TTS_PROBE_TEXT = "Hello world."

Origin = Literal["provider", "runner"]


@dataclass(frozen=True)
class ProbeSummary:
    scheduled_at: datetime
    probed: int
    failed: int


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
    try:
        result = await asyncio.wait_for(request(provider), timeout=timeout_s)
    except TimeoutError:
        return None, f"timed out after {timeout_s}s", "provider"
    except Exception as exc:
        return None, _truncate(str(exc) or type(exc).__name__), "provider"
    return result, None, None


def _excluded(entry: RegisteredModel, metric: Metric) -> bool:
    return (entry.provider, entry.model) in METRIC_EXCLUSIONS[metric]


def _to_ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000.0


async def _probe_stt(
    entry: RegisteredModel,
    item: Any,  # noqa: ANN401 — DatasetItem
    *,
    settings: Settings,
    gate: ModelGate,
    scheduled_at: datetime,
) -> ProbeResult:
    provider_cls = _get_stt_providers().get(entry.provider)
    pcm, _ = _trim_to_speech_end(
        _read_pcm16(item.path), item.duration_sec, item.speech_end_offset_ms
    )

    def row(sent_at: datetime, **fields: Any) -> ProbeResult:  # noqa: ANN401
        return ProbeResult(
            benchmark="STT",
            provider=entry.provider,
            model=entry.model,
            scheduled_at=scheduled_at,
            sent_at=sent_at,
            **fields,
        )

    if provider_cls is None:
        return row(datetime.now(UTC), error="unknown STT provider", failure_origin="runner")

    async with gate.slot(_gate_key(entry, provider_cls)):
        sent_at = datetime.now(UTC)
        result, error, origin = await _call(
            lambda: provider_cls(settings, entry.model),
            lambda provider: provider.measure_ttft(pcm, 1, 2, 16000, 0.1),
            _STT_TIMEOUT_S,
        )
    if result is not None:
        error = result.error or _stt_silent_failure(result)
        origin = "provider" if error else None
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
    settings: Settings,
    gate: ModelGate,
    scheduled_at: datetime,
) -> ProbeResult:
    provider_cls = _get_tts_providers().get(entry.provider)
    voice = entry.voices[0].id if entry.voices else entry.voice

    def row(sent_at: datetime, **fields: Any) -> ProbeResult:  # noqa: ANN401
        return ProbeResult(
            benchmark="TTS",
            provider=entry.provider,
            model=entry.model,
            voice=voice,
            scheduled_at=scheduled_at,
            sent_at=sent_at,
            **fields,
        )

    if provider_cls is None:
        return row(datetime.now(UTC), error="unknown TTS provider", failure_origin="runner")

    async with gate.slot(_gate_key(entry, provider_cls)):
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
            error, origin = result.error, "provider"
        elif result.ttfa_ms is None:
            error, origin = "no TTFA produced", "provider"
        else:
            # A cold or non-HTTP/2 connection isn't the provider being slow.
            error = _ttfa_transport_gate(result)
            origin = "runner" if error else None
    if result is None or error is not None:
        return row(sent_at, error=error or "no result", failure_origin=origin or "provider")
    return row(sent_at, ttfa_ms=result.ttfa_ms)


async def run_latency_probes(
    *,
    settings: Settings,
    kind: Literal["stt", "tts"],
    only: frozenset[str] | None = None,
) -> ProbeSummary:
    """Probe every collected shared *kind* model once and store each outcome."""
    scheduled_at = _current_tick(settings)
    async with lifespan_pool(settings) as pool:
        roster = await fetch_models(pool)
        stt_models, tts_models = _select_models(
            [m for m in roster if m.benchmark is Benchmark.STT],
            [m for m in roster if m.benchmark is Benchmark.TTS],
            source="shared",
            benchmark_kind=kind,
            only=only,
        )
        models = stt_models if kind == "stt" else tts_models
        gate = ModelGate(_CONCURRENCY_CAP)
        item = (
            load_stt_dataset(STT_PROBE_DATASET, settings=settings, sample_size=1).items[0]
            if kind == "stt" and models
            else None
        )

        async def probe(entry: RegisteredModel) -> ProbeResult:
            if kind == "stt":
                result = await _probe_stt(
                    entry, item, settings=settings, gate=gate, scheduled_at=scheduled_at
                )
            else:
                result = await _probe_tts(
                    entry, settings=settings, gate=gate, scheduled_at=scheduled_at
                )
            await insert_probe_result(pool, result)
            logger.info(
                "latency_probe",
                benchmark=result.benchmark,
                provider=result.provider,
                model=result.model,
                status=result.status,
                ttft_ms=result.ttft_ms,
                ttfs_ms=result.ttfs_ms,
                ttfa_ms=result.ttfa_ms,
                error=result.error,
            )
            return result

        try:
            await _warmup(models, settings)
            results = await asyncio.gather(*(probe(entry) for entry in models))
        finally:
            await _close_http_clients()
    return ProbeSummary(
        scheduled_at=scheduled_at,
        probed=len(results),
        failed=sum(result.error is not None for result in results),
    )
