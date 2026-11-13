# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality sweep: every item of every dataset through a model, once per model version.

A run opens a sweep for each model version and dataset version that has none,
which queues every item of the dataset. For TTS an item is a prompt in one of
the model's voices. It then sends each unfinished sweep's queued items to the
provider, one at a time. An item that gets a result leaves
the queue. One the provider fails loses a try and is run again on a later run
while it has tries left. A run stops starting new items when its time is up,
and the next run carries on from the queue.

Each provider call holds the model lock, the same one the latency probe takes.
A run also claims a model for as long as it works on it and skips one another
run has claimed, so several runs started together spread across models.
A clean transcript is scored for WER as soon as it's stored. For TTS the
transcript is Whisper's reading of the synthesized audio, and the audio is
kept in the artifact bucket.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import structlog

from coval_bench import gcs, telemetry
from coval_bench.config import require_capture_bucket
from coval_bench.datasets.loader import dataset_manifest, stt_item_fetcher
from coval_bench.datasets.suite import SWEEP_STT_DATASETS, SWEEP_TTS_DATASETS
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.model_lock import model_claim, model_lock, model_lock_key
from coval_bench.db.registry_store import RegistryStore, _registered
from coval_bench.db.sweep_store import (
    Score,
    Sweep,
    complete_sweep,
    open_sweeps,
    queued_count,
    queued_items,
    record_failure,
    store_result,
    store_score,
    unfinished_sweeps,
)
from coval_bench.metrics import compute_wer
from coval_bench.metrics.wer import NORM_VERSION
from coval_bench.providers._http_session import close_all as _close_http_clients
from coval_bench.registries import Benchmark, Metric
from coval_bench.runner.orchestrator import (
    _STT_TIMEOUT_S,
    _TTS_TIMEOUT_S,
    _get_stt_providers,
    _get_tts_providers,
    _read_pcm16,
    _select_models,
    _stt_silent_failure,
    _transcribe_with_whisper,
    _trim_to_speech_end,
    _truncate,
)

if TYPE_CHECKING:
    from coval_bench.config import Settings
    from coval_bench.datasets.manifest import Manifest, STTManifestItem, TTSManifestItem
    from coval_bench.registries import RegisteredModel

logger = structlog.get_logger("coval_bench.sweep")

Kind = Literal["stt", "tts"]


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
    audio_uri: str | None = None
    error: str | None = None


def _stt_items(manifest: Manifest) -> list[STTManifestItem]:
    return cast("list[STTManifestItem]", manifest.items)


@dataclass(frozen=True)
class _Item:
    sample_id: str
    voice: str
    reference: str
    raw: Any


def _items(kind: Kind, entry: RegisteredModel, manifest: Manifest) -> list[_Item]:
    """What a sweep of *manifest* has to run for *entry*; the voice is empty for STT."""
    if kind == "stt":
        return [
            _Item(item.sample_id or item.path, "", item.transcript, item)
            for item in _stt_items(manifest)
        ]
    voices = [voice.id for voice in entry.voices] or [entry.voice or ""]
    prompts = cast("list[TTSManifestItem]", manifest.items)
    return [
        _Item(prompt.testcase_id, voice, prompt.spoken_reference or prompt.transcript, prompt)
        for prompt in prompts
        for voice in voices
    ]


def _wer_score(reference: str, transcript: str) -> Score:
    wer = compute_wer(reference, transcript)
    return Score(
        metric=Metric.WER,
        metric_version=NORM_VERSION,
        errors=wer.substitutions + wer.deletions + wer.insertions,
        reference_units=wer.reference_words,
    )


async def _run_stt(
    entry: RegisteredModel,
    item: _Item,
    *,
    fetch: Any,  # noqa: ANN401 — the dataset's item fetcher
    settings: Settings,
    pool: Any,  # noqa: ANN401 — the connection pool
    log: Any,  # noqa: ANN401 — a bound structlog logger
) -> _Output | None:
    """Transcribe one clip. None when the item should just be left for the next run."""
    try:
        clip = await asyncio.to_thread(fetch, item.raw)
        pcm, _ = _trim_to_speech_end(
            _read_pcm16(clip.path), clip.duration_sec, clip.speech_end_offset_ms
        )
    except Exception:
        log.exception("sweep_item_not_loaded")
        return None
    try:
        provider_cls = _get_stt_providers()[entry.provider]
        key = model_lock_key(entry, provider_cls)
        async with model_lock(pool, key, timeout_s=_STT_TIMEOUT_S) as held:
            if not held:
                log.warning("sweep_item_skipped", reason="model busy")
                return None
            async with asyncio.timeout(_STT_TIMEOUT_S):
                result = await provider_cls(settings, entry.model).measure_ttft(
                    pcm, 1, 2, 16000, 0.1
                )
    except Exception as exc:
        return _Output(error=_truncate(str(exc) or type(exc).__name__))
    return _Output(
        transcript=result.complete_transcript,
        timings={
            "ttft_seconds": result.ttft_seconds,
            "audio_to_final_seconds": result.audio_to_final_seconds,
        },
        error=result.error or _stt_silent_failure(result),
    )


def _uploader(settings: Settings) -> Callable[[str, Path], str]:
    """A function that puts a file in the artifact bucket under a key and returns its URI."""
    require_capture_bucket(settings)
    bucket = gcs.client().bucket(settings.benchmark_artifact_bucket)

    def upload(key: str, path: Path) -> str:
        bucket.blob(key).upload_from_filename(str(path))
        return f"gs://{bucket.name}/{key}"

    return upload


async def _run_tts(
    entry: RegisteredModel,
    item: _Item,
    *,
    sweep_id: int,
    upload: Callable[[str, Path], str],
    settings: Settings,
    pool: Any,  # noqa: ANN401 — the connection pool
    log: Any,  # noqa: ANN401 — a bound structlog logger
) -> _Output | None:
    """Synthesize one prompt in one voice, transcribe the audio with Whisper and keep it.

    None when the item should just be left for the next run. A Whisper or upload
    failure counts against the item like a provider one: the provider has been paid by
    then, so an item that never ran out of tries would be paid for on every run. The
    local copy of the audio is always deleted.
    """
    try:
        provider_cls = _get_tts_providers()[entry.provider]
        key = model_lock_key(entry, provider_cls)
        async with model_lock(pool, key, timeout_s=_TTS_TIMEOUT_S) as held:
            if not held:
                log.warning("sweep_item_skipped", reason="model busy")
                return None
            async with asyncio.timeout(_TTS_TIMEOUT_S):
                result = await provider_cls(
                    settings=settings, model=entry.model, voice=item.voice or None
                ).synthesize(item.raw.transcript)
    except Exception as exc:
        return _Output(error=_truncate(str(exc) or type(exc).__name__))
    timings = {"ttfa_ms": result.ttfa_ms}
    audio_path = result.audio_path
    try:
        if result.error:
            return _Output(timings=timings, error=result.error)
        if audio_path is None or not audio_path.exists():
            return _Output(timings=timings, error="no audio produced")
        try:
            transcript = await _transcribe_with_whisper(audio_path, settings)
        except Exception as exc:
            return _Output(timings=timings, error=_truncate(f"whisper: {exc}"))
        key = f"sweeps/{sweep_id}/{item.sample_id}/{item.voice}{audio_path.suffix}"
        try:
            audio_uri = await asyncio.to_thread(upload, key, audio_path)
        except Exception as exc:
            return _Output(timings=timings, error=_truncate(f"upload: {exc}"))
        return _Output(transcript=transcript, timings=timings, audio_uri=audio_uri)
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)


async def _work_sweep(
    kind: Kind,
    entry: RegisteredModel,
    sweep: Sweep,
    manifest: Manifest,
    *,
    settings: Settings,
    pool: Any,  # noqa: ANN401 — the connection pool
    deadline: float,
    summary: SweepSummary,
) -> bool:
    """Run a sweep's queued items until they're done or time is up. False when time ran out."""
    if kind == "stt":
        fetch = stt_item_fetcher(sweep.dataset_id, settings=settings)
        run = partial(_run_stt, entry, fetch=fetch, settings=settings, pool=pool)
    else:
        run = partial(
            _run_tts,
            entry,
            sweep_id=sweep.id,
            upload=_uploader(settings),
            settings=settings,
            pool=pool,
        )
    queued = await queued_items(pool, sweep.id)
    sweep_log = logger.bind(
        provider=entry.provider,
        model=entry.model,
        dataset=sweep.dataset_id,
        sweep_id=sweep.id,
    )
    for item in _items(kind, entry, manifest):
        if (item.sample_id, item.voice) not in queued:
            continue
        if asyncio.get_running_loop().time() >= deadline:
            return False
        log = sweep_log.bind(sample_id=item.sample_id, voice=item.voice)
        output = await run(item, log=log)
        if output is None:
            continue
        summary.called += 1
        whisper = output.error is not None and output.error.startswith("whisper:")
        telemetry.record_sweep_item(
            kind=kind,
            provider="openai" if whisper else entry.provider,
            model="whisper-1" if whisper else entry.model,
            outcome=telemetry.outcome(
                output.error, ours=whisper or (output.error or "").startswith("upload:")
            ),
        )
        if output.error is not None:
            summary.failed += 1
            await record_failure(
                pool,
                sweep_id=sweep.id,
                sample_id=item.sample_id,
                voice=item.voice,
                error=output.error,
            )
            log.warning("sweep_item_failed", error=output.error)
            continue
        result_id = await store_result(
            pool,
            sweep_id=sweep.id,
            sample_id=item.sample_id,
            voice=item.voice,
            transcript=output.transcript,
            timings=output.timings,
            audio_uri=output.audio_uri,
        )
        if result_id is not None and output.transcript is not None:
            try:
                score = _wer_score(item.reference, output.transcript)
            except Exception:
                log.exception("sweep_item_not_scored")
            else:
                await store_score(pool, result_id, score)
                summary.scored += 1
    if await complete_sweep(pool, sweep.id):
        summary.completed += 1
        sweep_log.info("sweep_completed", items=len(manifest.items))
    return True


async def run_sweeps(
    *,
    settings: Settings,
    kind: Kind,
    only: frozenset[str] | None = None,
    max_minutes: float,
) -> SweepSummary:
    """Open any missing sweeps for the collected shared *kind* models, then work on them."""
    benchmark = Benchmark.STT if kind == "stt" else Benchmark.TTS
    datasets = SWEEP_STT_DATASETS if kind == "stt" else SWEEP_TTS_DATASETS
    summary = SweepSummary()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_minutes * 60
    async with lifespan_pool(settings) as pool:
        records = [r for r in await RegistryStore(pool).list_models() if r.modality is benchmark]
        ids = {(r.provider, r.model): r.id for r in records}
        roster = [_registered(r) for r in records]
        stt_models, tts_models = _select_models(
            roster if kind == "stt" else [],
            roster if kind == "tts" else [],
            source="shared",
            benchmark_kind=kind,
            only=only,
        )
        models = stt_models if kind == "stt" else tts_models
        entries = {ids[(m.provider, m.model)]: m for m in models}
        manifests = {dataset_id: dataset_manifest(dataset_id) for dataset_id in datasets}
        summary.models = len(models)
        summary.opened = await open_sweeps(
            pool,
            [
                (
                    model_id,
                    dataset_id,
                    manifest.version,
                    [(item.sample_id, item.voice) for item in _items(kind, entry, manifest)],
                )
                for model_id, entry in entries.items()
                for dataset_id, manifest in manifests.items()
            ],
        )

        providers = _get_stt_providers() if kind == "stt" else _get_tts_providers()
        try:
            by_claim: dict[str, list[Sweep]] = {}
            for sweep in await unfinished_sweeps(pool, list(entries)):
                if manifests[sweep.dataset_id].version != sweep.dataset_version:
                    continue
                entry = entries[sweep.model_id]
                key = model_lock_key(entry, providers.get(entry.provider))
                by_claim.setdefault(key, []).append(sweep)
            in_time = True
            for key, sweeps in by_claim.items():
                if not in_time:
                    break
                async with model_claim(pool, key) as claimed:
                    if not claimed:
                        logger.info("sweep_model_skipped", claim=key, reason="another run has it")
                        continue
                    for sweep in sweeps:
                        in_time = await _work_sweep(
                            kind,
                            entries[sweep.model_id],
                            sweep,
                            manifests[sweep.dataset_id],
                            settings=settings,
                            pool=pool,
                            deadline=deadline,
                            summary=summary,
                        )
                        if not in_time:
                            break
            sweep_ids = [sweep.id for sweeps in by_claim.values() for sweep in sweeps]
            telemetry.record_sweep_queued(kind=kind, queued=await queued_count(pool, sweep_ids))
        finally:
            await _close_http_clients()
    return summary
