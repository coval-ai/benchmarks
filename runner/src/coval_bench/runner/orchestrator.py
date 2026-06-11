# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Main benchmark run loop.

``run_benchmarks`` is the function the Cloud Run Job invokes once per scheduled
trigger. It wires dataset loading → parallel provider dispatch with per-call
timeout + retry → metric computation → per-task DB persistence → run summary.

Persistence is per-task (each ``_run_stt_item`` / ``_run_tts_item`` flushes
its own results before returning) rather than a single batch at end-of-run,
so a job-timeout mid-run still durably stores everything that completed.

Design notes
------------
- Sibling modules (providers, datasets, db, metrics) are imported lazily via
  ``importlib.import_module`` so that ``coval_bench.runner`` is importable on
  its own even when sibling agents haven't landed yet.  Names are resolved at
  call time, not at module load, which also lets tests patch ``sys.modules``
  without triggering import errors.  ``coval_bench.registries`` is the
  exception: dependency-light by design, imported eagerly.
- Concurrency: ``asyncio.Semaphore(8)`` caps simultaneous provider connections.
- Timeouts: ``asyncio.timeout(45)`` for STT, ``asyncio.timeout(60)`` for TTS.
- Audio cleanup: TTS audio files are deleted in ``finally`` blocks; this module
  owns cleanup, NOT the provider.
- Status logic: 100% success → SUCCEEDED, 0% → FAILED, otherwise PARTIAL.
- Structured logging: all lines via ``structlog.get_logger("coval_bench.runner")``.
  The exact string ``event="RUN_FAILED"`` in an ERROR log is the trigger for the
  ``benchmark_run_failure`` Cloud Logging metric (Cloud Logging severity ERROR +
  literal ``RUN_FAILED``).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import hashlib
import importlib
import signal
from datetime import UTC, datetime  # noqa: UP017 — UTC alias requires 3.11+, target is 3.12
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog
from posthog import Posthog
from pydantic import BaseModel

from coval_bench.providers._http_session import close_all as _close_http_clients
from coval_bench.providers.base import Provider
from coval_bench.registries import (
    METRIC_SPECS,
    MODEL_REGISTRY,
    Benchmark,
    Metric,
    ModelStatus,
    RegisteredModel,
)
from coval_bench.runner.retry import with_retry

if TYPE_CHECKING:
    from coval_bench.config import Settings

_log = structlog.get_logger("coval_bench.runner")

_CONCURRENCY_CAP = 8
_STT_TIMEOUT_S = 45
_TTS_TIMEOUT_S = 60
_MAX_ERROR_LEN = 4000  # truncate error messages stored in DB (Postgres text is unbounded
# but huge stack traces choke log pipelines)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Typed return value from ``run_benchmarks``."""

    run_id: int
    started_at: datetime
    finished_at: datetime
    status: str  # RunStatus StrEnum value — avoids hard import at module load
    total_results: int
    success_count: int
    fail_count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truncate(msg: str) -> str:
    """Truncate error messages to ``_MAX_ERROR_LEN`` chars."""
    return msg[:_MAX_ERROR_LEN]


def _metric_outcome(
    metric_value: float | None,
    item_error: str | None,
    metric_label: str,
    result_status: Any,  # noqa: ANN401 — ResultStatus enum, lazy-imported by callers
) -> tuple[Any, str | None]:
    """Decide ``(status, error)`` for a single result row.

    A provider that returns without raising can still have failed: it may set
    ``result.error`` or simply produce no measurement. The orchestrator used to
    stamp every returned row ``SUCCESS``; this restores honest labelling.

    Rules (in order):
    * an item-level provider error fails the row and carries the message;
    * otherwise a missing (``None``) metric fails just this row;
    * a present metric succeeds.
    """
    if item_error:
        return result_status.FAILED, _truncate(item_error)
    if metric_value is None:
        return result_status.FAILED, f"no {metric_label} produced"
    return result_status.SUCCESS, None


async def _transcribe_with_whisper(audio_path: Path, settings: Settings) -> str:
    """Transcribe *audio_path* with OpenAI ``whisper-1`` and return the text.

    Used by the TTS path to compute WER against the synthesized audio.
    Raises on API error — caller handles and records as a failed Result.
    """
    import openai

    api_key = settings.openai_api_key
    if api_key is None:
        raise RuntimeError("openai_api_key is required for TTS-WER computation")

    client = openai.AsyncOpenAI(api_key=api_key.get_secret_value())
    with audio_path.open("rb") as fh:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=fh,
        )
    return str(response.text)


def _get_stt_providers() -> dict[str, Any]:
    """Resolve the STT provider registry at call time (lazy import)."""
    mod = importlib.import_module("coval_bench.providers.stt")
    return mod.STT_PROVIDERS  # type: ignore[no-any-return]


def _get_tts_providers() -> dict[str, Any]:
    """Resolve the TTS provider registry at call time (lazy import)."""
    mod = importlib.import_module("coval_bench.providers.tts")
    return mod.TTS_PROVIDERS  # type: ignore[no-any-return]


def _get_load_dataset() -> Any:  # noqa: ANN401
    """Resolve ``load_dataset`` at call time (lazy import)."""
    mod = importlib.import_module("coval_bench.datasets")
    return mod.load_dataset


def _get_db_symbols() -> tuple[Any, Any, Any, Any]:
    """Return (lifespan_pool, RunWriter, RunStatus, ResultStatus, Result) at call time."""
    conn_mod = importlib.import_module("coval_bench.db.conn")
    writer_mod = importlib.import_module("coval_bench.db.writer")
    models_mod = importlib.import_module("coval_bench.db.models")
    return (
        conn_mod.lifespan_pool,
        writer_mod.RunWriter,
        models_mod.RunStatus,
        models_mod,
    )


def _get_metrics() -> tuple[Any, Any]:
    """Return (compute_wer, compute_rtf) at call time."""
    mod = importlib.import_module("coval_bench.metrics")
    return mod.compute_wer, mod.compute_rtf


# ---------------------------------------------------------------------------
# STT coroutine builder
# ---------------------------------------------------------------------------


async def _run_stt_item(
    *,
    entry: RegisteredModel,
    item: Any,  # noqa: ANN401 — DatasetItem is a runtime-typed sibling-agent type
    run_id: int,
    sem: asyncio.Semaphore,
    settings: Settings,
    writer: Any | None = None,  # noqa: ANN401 — RunWriter, lazy-imported in caller
) -> list[Any]:
    """Run a single STT provider × dataset item, returning a list of Result rows.

    If *writer* is non-None, results are persisted before returning. This is the
    incremental-flush path used by ``run_benchmarks``: a task that completes is
    durably stored even if a sibling task or the whole job is later cancelled.
    """
    stt_providers = _get_stt_providers()
    _, _, _, models_mod = _get_db_symbols()
    Benchmark = models_mod.Benchmark
    Result = models_mod.Result
    ResultStatus = models_mod.ResultStatus
    compute_wer, compute_rtf = _get_metrics()

    results: list[Any] = []

    async with sem:
        provider_cls = stt_providers.get(entry.provider)
        if provider_cls is None:
            _log.warning(
                "unknown STT provider — skipping",
                provider=entry.provider,
                model=entry.model,
            )
            return []

        key_attr = f"{entry.provider}_api_key"
        api_key = getattr(settings, key_attr, None)
        kwargs: dict[str, Any] = {"api_key": api_key, "model": entry.model}
        if entry.provider == "google":
            kwargs["project_id"] = settings.google_project_id
        provider = provider_cls(**kwargs)

        audio_path: Path = item.path
        transcript_ref: str = item.transcript
        duration_sec: float = item.duration_sec
        audio_bytes = audio_path.read_bytes()

        speech_end_offset_ms = item.speech_end_offset_ms
        if isinstance(speech_end_offset_ms, (int, float)):
            trailing_ms = max(0.0, duration_sec * 1000.0 - speech_end_offset_ms)
            tail_bytes = int(round(trailing_ms / 1000.0 * 16000)) * 2
            if 0 < tail_bytes < len(audio_bytes):
                audio_bytes = audio_bytes[: len(audio_bytes) - tail_bytes]
                # Audio now ends at speech-end; reflect the trimmed length so RTF and the
                # provider duration hint match what was actually sent.
                duration_sec = speech_end_offset_ms / 1000.0

        transcription_result = None
        item_error: str | None = None
        try:
            async with asyncio.timeout(_STT_TIMEOUT_S):
                transcription_result = await with_retry(
                    lambda: provider.measure_ttft(
                        audio_bytes,
                        1,  # mono
                        2,  # PCM_16 = 2 bytes/sample
                        16000,
                        0.1,
                    ),
                )
        except Exception as exc:
            item_error = _truncate(str(exc))
            _log.warning(
                "STT provider call failed",
                provider=entry.provider,
                model=entry.model,
                audio=str(audio_path),
                exc_info=exc,
            )

        item_error = item_error or (
            transcription_result.error if transcription_result is not None else None
        )
        ttft_seconds = transcription_result.ttft_seconds if transcription_result else None
        audio_to_final = (
            transcription_result.audio_to_final_seconds if transcription_result else None
        )
        complete_transcript = (
            transcription_result.complete_transcript if transcription_result else None
        )

        # 1. TTFT
        ttft_status, ttft_error = _metric_outcome(
            ttft_seconds, item_error, Metric.TTFT, ResultStatus
        )
        results.append(
            Result(
                run_id=run_id,
                provider=entry.provider,
                model=entry.model,
                benchmark=Benchmark.STT,
                metric_type=Metric.TTFT,
                metric_value=ttft_seconds,
                metric_units=METRIC_SPECS[Metric.TTFT].units,
                audio_filename=audio_path.name,
                transcript=complete_transcript,
                status=ttft_status,
                error=ttft_error,
            )
        )

        # 2. AudioToFinal
        atf_status, atf_error = _metric_outcome(
            audio_to_final, item_error, Metric.AUDIO_TO_FINAL, ResultStatus
        )
        results.append(
            Result(
                run_id=run_id,
                provider=entry.provider,
                model=entry.model,
                benchmark=Benchmark.STT,
                metric_type=Metric.AUDIO_TO_FINAL,
                metric_value=audio_to_final,
                metric_units=METRIC_SPECS[Metric.AUDIO_TO_FINAL].units,
                audio_filename=audio_path.name,
                transcript=complete_transcript,
                status=atf_status,
                error=atf_error,
            )
        )

        # 2b. TTFS — time-to-final from VAD end-of-speech (primary headline metric). Status
        # tracks ttfs_value (not audio_to_final): a missing offset or a failed calculation
        # fails the row instead of passing as a null-valued success, and calc errors surface.
        ttfs_value: float | None = None
        ttfs_calc_error: str | None = None
        if audio_to_final is not None and isinstance(speech_end_offset_ms, (int, float)):
            try:
                compute_ttfs = importlib.import_module("coval_bench.metrics").compute_ttfs
                ttfs_value = compute_ttfs(audio_to_final, speech_end_offset_ms / 1000.0)
            except Exception as exc:
                ttfs_calc_error = str(exc)
        ttfs_status, ttfs_error = _metric_outcome(
            ttfs_value, item_error or ttfs_calc_error, Metric.TTFS, ResultStatus
        )
        # Flux can't finalize on our signal (no Finalize, EOT can't be disabled), so it's
        # outside the TTFS parity cohort. Other metrics still run; only the TTFS row is omitted.
        ttfs_excluded = entry.provider == "deepgram" and entry.model.startswith("flux-")
        if not ttfs_excluded:
            results.append(
                Result(
                    run_id=run_id,
                    provider=entry.provider,
                    model=entry.model,
                    benchmark=Benchmark.STT,
                    metric_type=Metric.TTFS,
                    metric_value=ttfs_value,
                    metric_units=METRIC_SPECS[Metric.TTFS].units,
                    audio_filename=audio_path.name,
                    transcript=complete_transcript,
                    status=ttfs_status,
                    error=ttfs_error,
                )
            )

        # 3. RTF — derived from AudioToFinal, so its outcome tracks whether a final was
        # produced. A present final with an uncomputable RTF (e.g. zero duration) stays a
        # null-valued success rather than a failure, matching the original intent.
        rtf_value: float | None = None
        if audio_to_final is not None and duration_sec > 0:
            with contextlib.suppress(Exception):
                rtf_value = compute_rtf(audio_to_final, duration_sec)

        rtf_status, rtf_error = _metric_outcome(
            audio_to_final, item_error, Metric.RTF, ResultStatus
        )
        results.append(
            Result(
                run_id=run_id,
                provider=entry.provider,
                model=entry.model,
                benchmark=Benchmark.STT,
                metric_type=Metric.RTF,
                metric_value=rtf_value,
                metric_units=METRIC_SPECS[Metric.RTF].units,
                audio_filename=audio_path.name,
                transcript=complete_transcript,
                status=rtf_status,
                error=rtf_error,
            )
        )

        # 4. WER (when ground-truth available; skip when the stream errored so we don't
        # score a transcript salvaged from a failed run)
        if transcript_ref and complete_transcript is not None and item_error is None:
            try:
                wer_result = compute_wer(transcript_ref, complete_transcript)
                results.append(
                    Result(
                        run_id=run_id,
                        provider=entry.provider,
                        model=entry.model,
                        benchmark=Benchmark.STT,
                        metric_type=Metric.WER,
                        metric_value=wer_result.wer_percentage,
                        metric_units=METRIC_SPECS[Metric.WER].units,
                        audio_filename=audio_path.name,
                        transcript=complete_transcript,
                        status=ResultStatus.SUCCESS,
                        error=None,
                    )
                )
            except Exception as exc:
                # compute_wer is deterministic over a transcript we already obtained, so a
                # crash here is a genuine failure — surface it as a FAILED row rather than
                # silently dropping it and leaving the run marked SUCCEEDED.
                _log.warning(
                    "WER computation failed",
                    provider=entry.provider,
                    model=entry.model,
                    exc_info=exc,
                )
                results.append(
                    Result(
                        run_id=run_id,
                        provider=entry.provider,
                        model=entry.model,
                        benchmark=Benchmark.STT,
                        metric_type=Metric.WER,
                        metric_value=None,
                        metric_units=None,
                        audio_filename=audio_path.name,
                        transcript=complete_transcript,
                        status=ResultStatus.FAILED,
                        error=_truncate(str(exc)),
                    )
                )

    if writer is not None and results:
        try:
            await writer.record_results(results)
        except Exception as exc:
            _log.warning(
                "STT result persist failed",
                provider=entry.provider,
                model=entry.model,
                exc_info=exc,
            )

    return results


# ---------------------------------------------------------------------------
# TTS coroutine builder
# ---------------------------------------------------------------------------


async def _run_tts_item(
    *,
    entry: RegisteredModel,
    item: Any,  # noqa: ANN401 — TTSDatasetItem is a runtime-typed sibling-agent type
    run_id: int,
    sem: asyncio.Semaphore,
    settings: Settings,
    writer: Any | None = None,  # noqa: ANN401 — RunWriter, lazy-imported in caller
) -> list[Any]:
    """Run a single TTS provider × dataset item, returning a list of Result rows.

    If *writer* is non-None, results are persisted before returning. See
    :func:`_run_stt_item` for the rationale (incremental flush).
    """
    tts_providers = _get_tts_providers()
    _, _, _, models_mod = _get_db_symbols()
    Benchmark = models_mod.Benchmark
    Result = models_mod.Result
    ResultStatus = models_mod.ResultStatus
    compute_wer, _ = _get_metrics()

    results: list[Any] = []
    audio_path: Path | None = None

    async with sem:
        provider_cls = tts_providers.get(entry.provider)
        if provider_cls is None:
            _log.warning(
                "unknown TTS provider — skipping",
                provider=entry.provider,
                model=entry.model,
            )
            return []

        transcript: str = item.transcript
        provider = provider_cls(settings=settings, model=entry.model, voice=entry.voice)

        tts_result = None
        item_error: str | None = None
        try:
            async with asyncio.timeout(_TTS_TIMEOUT_S):
                tts_result = await with_retry(
                    lambda: provider.synthesize(transcript),
                )
            audio_path = tts_result.audio_path
        except Exception as exc:
            item_error = _truncate(str(exc))
            _log.warning(
                "TTS provider call failed",
                provider=entry.provider,
                model=entry.model,
                exc_info=exc,
            )

        # Single failure model: a raised exception OR a provider-reported error fails the
        # TTFA row (see _metric_outcome) and suppresses WER scoring below.
        item_error = item_error or (tts_result.error if tts_result is not None else None)
        ttfa_ms = tts_result.ttfa_ms if tts_result else None

        try:
            # 1. TTFA
            ttfa_status, ttfa_error = _metric_outcome(
                ttfa_ms, item_error, Metric.TTFA, ResultStatus
            )
            ttfa_value = ttfa_ms

            # Transport-contamination gate: a clean measurement taken over HTTP/1.1 or a cold
            # socket is not comparable to the WebSocket cohort. Override only a would-be
            # SUCCESS — a real provider error or empty result keeps its own message.
            if ttfa_status is ResultStatus.SUCCESS and tts_result is not None:
                if tts_result.http_version is not None and tts_result.http_version != "HTTP/2":
                    ttfa_status = ResultStatus.FAILED
                    ttfa_error = _truncate(
                        f"TTFA measured over {tts_result.http_version}; not comparable "
                        "(no HTTP/2 multiplexing, TTFA reabsorbs TCP+TLS)"
                    )
                    ttfa_value = None
                elif tts_result.connection_reused is False:
                    ttfa_status = ResultStatus.FAILED
                    ttfa_error = _truncate(
                        "TTFA measured over a cold connection; not comparable "
                        "(TTFA reabsorbs TCP+TLS)"
                    )
                    ttfa_value = None

            results.append(
                Result(
                    run_id=run_id,
                    provider=entry.provider,
                    model=entry.model,
                    voice=entry.voice,
                    benchmark=Benchmark.TTS,
                    metric_type=Metric.TTFA,
                    metric_value=ttfa_value,
                    metric_units=METRIC_SPECS[Metric.TTFA].units,
                    audio_filename=audio_path.name if audio_path else None,
                    transcript=transcript,
                    status=ttfa_status,
                    error=ttfa_error,
                    http_version=tts_result.http_version if tts_result else None,
                    submit_to_headers_ms=tts_result.submit_to_headers_ms if tts_result else None,
                )
            )

            # 2. WER via Whisper transcription of synthesized audio (skip when synth errored)
            if item_error is None and audio_path is not None and audio_path.exists():
                # Whisper transcription is our measurement instrument, not the provider under
                # test — a transcription failure is logged and skipped (no WER row), so a
                # transient Whisper outage doesn't drag the provider's run to PARTIAL.
                whisper_transcript: str | None = None
                try:
                    whisper_transcript = await _transcribe_with_whisper(audio_path, settings)
                except Exception as exc:
                    _log.warning(
                        "TTS WER transcription failed",
                        provider=entry.provider,
                        model=entry.model,
                        exc_info=exc,
                    )

                # compute_wer is deterministic; if it raises on a real transcript, that's a
                # genuine failure worth surfacing as a FAILED row rather than silently dropping.
                if whisper_transcript is not None:
                    try:
                        wer_result = compute_wer(transcript, whisper_transcript)
                        results.append(
                            Result(
                                run_id=run_id,
                                provider=entry.provider,
                                model=entry.model,
                                voice=entry.voice,
                                benchmark=Benchmark.TTS,
                                metric_type=Metric.WER,
                                metric_value=wer_result.wer_percentage,
                                metric_units=METRIC_SPECS[Metric.WER].units,
                                audio_filename=audio_path.name,
                                transcript=whisper_transcript,
                                status=ResultStatus.SUCCESS,
                                error=None,
                            )
                        )
                    except Exception as exc:
                        _log.warning(
                            "TTS WER computation failed",
                            provider=entry.provider,
                            model=entry.model,
                            exc_info=exc,
                        )
                        results.append(
                            Result(
                                run_id=run_id,
                                provider=entry.provider,
                                model=entry.model,
                                voice=entry.voice,
                                benchmark=Benchmark.TTS,
                                metric_type=Metric.WER,
                                metric_value=None,
                                metric_units=None,
                                audio_filename=audio_path.name,
                                transcript=whisper_transcript,
                                status=ResultStatus.FAILED,
                                error=_truncate(str(exc)),
                            )
                        )
        finally:
            # Orchestrator owns audio cleanup — always delete in finally block
            if audio_path is not None and audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError as exc:
                    _log.warning(
                        "failed to delete TTS audio file",
                        path=str(audio_path),
                        exc_info=exc,
                    )

    if writer is not None and results:
        try:
            await writer.record_results(results)
        except Exception as exc:
            _log.warning(
                "TTS result persist failed",
                provider=entry.provider,
                model=entry.model,
                exc_info=exc,
            )

    return results


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def _emit_posthog(client: Posthog | None, event: str, properties: dict[str, Any]) -> None:
    """Best-effort run-event capture + flush; never raises into the run outcome."""
    if client is None:
        return
    try:
        client.capture(
            event, distinct_id="coval-bench-runner", properties=properties, disable_geoip=True
        )
        client.flush()  # type: ignore[no-untyped-call]
    except Exception:
        _log.warning("posthog_emit_failed", event_name=event, exc_info=True)


async def run_benchmarks(
    *,
    settings: Settings,
    benchmark_kind: Literal["stt", "tts", "both"] = "both",
    smoke: bool = False,
    matrix_overrides: list[RegisteredModel] | None = None,
) -> RunSummary:
    """Execute one complete benchmark run.

    Args:
        settings: Injected application settings (no global state).
        benchmark_kind: Which benchmark(s) to run.
        smoke: If True, process only the first dataset item (local dev mode).
        matrix_overrides: Optional list of ``RegisteredModel`` objects that
            override the registry by ``(benchmark, provider, model)`` key.

    Returns:
        A :class:`RunSummary` with final counts and run status.

    Raises:
        Exception: On unrecoverable errors (dataset integrity failure, DB issues).
            The run row is updated to ``FAILED`` before re-raising so the Cloud
            Run Job exits non-zero and the log-based metric fires.
    """
    lifespan_pool, RunWriter, RunStatus, models_mod = _get_db_symbols()
    Result = models_mod.Result
    ResultStatus = models_mod.ResultStatus
    load_dataset = _get_load_dataset()

    posthog_client: Posthog | None = None
    if not settings.posthog_disabled and settings.posthog_project_token:
        try:
            posthog_client = Posthog(
                settings.posthog_project_token,
                host=settings.posthog_host,
                enable_exception_autocapture=True,
            )
            atexit.register(posthog_client.shutdown)
        except Exception:
            _log.warning("posthog_init_failed", exc_info=True)
            posthog_client = None

    # ------------------------------------------------------------------
    # 1. Resolve + filter the model registry
    # ------------------------------------------------------------------
    stt_matrix = [m for m in MODEL_REGISTRY if m.benchmark is Benchmark.STT]
    tts_matrix = [m for m in MODEL_REGISTRY if m.benchmark is Benchmark.TTS]

    if matrix_overrides:
        override_map: dict[tuple[Benchmark, str, str], RegisteredModel] = {
            (e.benchmark, e.provider, e.model): e for e in matrix_overrides
        }
        stt_matrix = [override_map.get((e.benchmark, e.provider, e.model), e) for e in stt_matrix]
        tts_matrix = [override_map.get((e.benchmark, e.provider, e.model), e) for e in tts_matrix]

        # Also apply any overrides that add *new* models
        existing_keys = {(e.benchmark, e.provider, e.model) for e in stt_matrix + tts_matrix}
        for ov in matrix_overrides:
            if (ov.benchmark, ov.provider, ov.model) not in existing_keys:
                (stt_matrix if ov.benchmark is Benchmark.STT else tts_matrix).append(ov)

    enabled_stt = [e for e in stt_matrix if e.status is ModelStatus.ACTIVE]
    enabled_tts = [e for e in tts_matrix if e.status is ModelStatus.ACTIVE]

    # ------------------------------------------------------------------
    # 2. Open DB pool + start run row
    # ------------------------------------------------------------------
    async with lifespan_pool(settings) as pool:
        writer = RunWriter(pool)

        # Dataset SHA256 for the run record (computed from the packaged manifest)
        try:
            import importlib.resources as _importlib_resources

            manifest_ref = _importlib_resources.files("coval_bench.datasets.manifests").joinpath(
                "stt-v1.json"
            )
            manifest_bytes = manifest_ref.read_bytes()
            dataset_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        except Exception:
            dataset_sha256 = "unknown"

        period = settings.schedule_period_seconds
        epoch = int(datetime.now(tz=UTC).timestamp())
        scheduled_at = datetime.fromtimestamp(epoch - epoch % period, tz=UTC)
        run = await writer.start_run(
            runner_sha=settings.runner_sha,
            dataset_id=settings.dataset_id,
            dataset_sha256=dataset_sha256,
            scheduled_at=scheduled_at,
        )
        run_id = run.id
        assert run_id is not None  # noqa: S101 — DB always returns an id after INSERT
        started_at = run.started_at or datetime.now(tz=UTC)

        _log.info(
            "benchmark run started",
            run_id=run_id,
            benchmark_kind=benchmark_kind,
            smoke=smoke,
            runner_sha=settings.runner_sha,
        )

        all_results: list[Any] = []
        sem = asyncio.Semaphore(_CONCURRENCY_CAP)

        # Cloud Run sends SIGTERM ~10s before SIGKILL when a task hits its timeout.
        # We catch it, cancel the in-flight gather, and finalize the run row as
        # PARTIAL so timed-out executions surface in /v1/results instead of
        # remaining stuck at status='running' (invisible to the API filter).
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        sigterm_received = False

        def _on_sigterm() -> None:
            nonlocal sigterm_received
            if sigterm_received:
                return
            sigterm_received = True
            _log.warning("SIGTERM received — finalizing run as partial", run_id=run_id)
            if main_task is not None:
                main_task.cancel()

        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal.SIGTERM, _on_sigterm)

        try:
            # ------------------------------------------------------------------
            # 2b. Provider warmup
            # Each provider class may override Provider.warmup() to absorb
            # pre-t0 deployment cold-start cost (HTTP TCP+TLS, gRPC channel
            # open, NVCF dispatch).  Default is a no-op; only providers that
            # need it pay the call.  Errors are non-fatal.
            # ------------------------------------------------------------------
            stt_providers = _get_stt_providers()
            tts_providers = _get_tts_providers()
            provider_classes: set[type[Provider]] = set()
            if benchmark_kind in ("stt", "both"):
                for entry in enabled_stt:
                    cls = stt_providers.get(entry.provider)
                    if cls is not None:
                        provider_classes.add(cls)
            if benchmark_kind in ("tts", "both"):
                for entry in enabled_tts:
                    cls = tts_providers.get(entry.provider)
                    if cls is not None:
                        provider_classes.add(cls)
            if provider_classes:
                ordered_classes = list(provider_classes)
                warmup_results = await asyncio.gather(
                    *(cls.warmup(settings=settings) for cls in ordered_classes),
                    return_exceptions=True,
                )
                for cls, res in zip(ordered_classes, warmup_results, strict=False):
                    if isinstance(res, BaseException):
                        _log.warning(
                            "provider_warmup_failed",
                            provider=cls.__name__,
                            exc_info=res,
                        )

            # ------------------------------------------------------------------
            # 3. STT path
            # ------------------------------------------------------------------
            if benchmark_kind in ("stt", "both") and enabled_stt:
                stt_dataset = load_dataset("stt-v1", settings=settings)
                items = stt_dataset.items[:1] if smoke else stt_dataset.items

                stt_tasks = [
                    _run_stt_item(
                        entry=entry,
                        item=item,
                        run_id=run_id,
                        sem=sem,
                        settings=settings,
                        writer=writer,
                    )
                    for item in items
                    for entry in enabled_stt
                ]

                stt_batch = await asyncio.gather(*stt_tasks, return_exceptions=True)
                for batch_result in stt_batch:
                    if isinstance(batch_result, BaseException):
                        _log.warning("STT task raised unexpectedly", exc_info=batch_result)
                    else:
                        all_results.extend(batch_result)

            # ------------------------------------------------------------------
            # 4. TTS path
            # ------------------------------------------------------------------
            if benchmark_kind in ("tts", "both") and enabled_tts:
                tts_dataset = load_dataset("tts-v1", settings=settings)
                tts_items = tts_dataset.items[:1] if smoke else tts_dataset.items

                tts_tasks = [
                    _run_tts_item(
                        entry=entry,
                        item=item,
                        run_id=run_id,
                        sem=sem,
                        settings=settings,
                        writer=writer,
                    )
                    for item in tts_items
                    for entry in enabled_tts
                ]

                tts_batch = await asyncio.gather(*tts_tasks, return_exceptions=True)
                for batch_result in tts_batch:
                    if isinstance(batch_result, BaseException):
                        _log.warning("TTS task raised unexpectedly", exc_info=batch_result)
                    else:
                        all_results.extend(batch_result)

            # ------------------------------------------------------------------
            # 5. Compute final run status
            # ------------------------------------------------------------------
            # Results were persisted incrementally inside _run_stt_item /
            # _run_tts_item — see those functions' ``writer`` argument. This
            # ensures a job-timeout mid-run still durably stores everything
            # that completed, instead of losing the whole run's results in a
            # rolled-back batch INSERT.
            typed_results = [r for r in all_results if isinstance(r, Result)]
            success_count = sum(1 for r in typed_results if r.status == ResultStatus.SUCCESS)
            fail_count = sum(1 for r in typed_results if r.status == ResultStatus.FAILED)
            total_results = len(typed_results)

            if total_results == 0 or fail_count == total_results:
                final_status = RunStatus.FAILED
            elif success_count == total_results:
                final_status = RunStatus.SUCCEEDED
            else:
                final_status = RunStatus.PARTIAL

            await writer.finish_run(run_id, status=final_status, error=None)

            # Refresh after finish_run: the view query only counts runs already
            # marked succeeded/partial. A failed refresh must not fail the run.
            try:
                await writer.refresh_stats_matviews()
            except Exception as refresh_exc:
                _log.error(
                    "failed to refresh stats matviews",
                    run_id=run_id,
                    exc_info=refresh_exc,
                )

            finished_at = datetime.now(tz=UTC)
            summary = RunSummary(
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                status=str(final_status),
                total_results=total_results,
                success_count=success_count,
                fail_count=fail_count,
            )

            duration_s = (finished_at - started_at).total_seconds()
            _log.info(
                "benchmark run finished",
                run_id=run_id,
                status=str(final_status),
                total_results=total_results,
                success_count=success_count,
                fail_count=fail_count,
                duration_s=duration_s,
            )
            _emit_posthog(
                posthog_client,
                "benchmark_run_completed",
                {
                    "status": str(final_status),
                    "total_results": total_results,
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "duration_s": duration_s,
                    "benchmark_kind": benchmark_kind,
                    "smoke": smoke,
                    "$process_person_profile": False,
                },
            )
            return summary

        except asyncio.CancelledError:
            # SIGTERM-driven cancellation: write PARTIAL so the run is visible
            # to /v1/results. Any other CancelledError (e.g., parent shutdown)
            # is re-raised so it propagates normally.
            if not sigterm_received:
                raise
            if main_task is not None:
                main_task.uncancel()
            typed_results = [r for r in all_results if isinstance(r, Result)]
            success_count = sum(1 for r in typed_results if r.status == ResultStatus.SUCCESS)
            fail_count = sum(1 for r in typed_results if r.status == ResultStatus.FAILED)
            total_results = len(typed_results)
            try:
                await asyncio.shield(
                    writer.finish_run(
                        run_id,
                        status=RunStatus.PARTIAL,
                        error="sigterm-received (cloud-run-task-timeout)",
                    )
                )
            except Exception as write_exc:
                _log.error(
                    "failed to update run row after SIGTERM",
                    run_id=run_id,
                    exc_info=write_exc,
                )
            finished_at = datetime.now(tz=UTC)
            sigterm_duration_s = (finished_at - started_at).total_seconds()
            _log.warning(
                "benchmark run finished early due to SIGTERM",
                run_id=run_id,
                status=str(RunStatus.PARTIAL),
                total_results=total_results,
                success_count=success_count,
                fail_count=fail_count,
                duration_s=sigterm_duration_s,
            )
            _emit_posthog(
                posthog_client,
                "benchmark_run_completed",
                {
                    "status": str(RunStatus.PARTIAL),
                    "total_results": total_results,
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "duration_s": sigterm_duration_s,
                    "benchmark_kind": benchmark_kind,
                    "smoke": smoke,
                    "sigterm": True,
                    "$process_person_profile": False,
                },
            )
            return RunSummary(
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                status=str(RunStatus.PARTIAL),
                total_results=total_results,
                success_count=success_count,
                fail_count=fail_count,
            )

        except Exception as exc:
            # Unrecoverable error — log RUN_FAILED (triggers Cloud Logging metric),
            # update run row, then re-raise so Cloud Run Job exits non-zero.
            err_msg = _truncate(str(exc))
            # The literal event="RUN_FAILED" in this log line triggers the
            # Cloud Logging log-based metric (see ARCHITECTURE.md § Logging + Alerting).
            # structlog uses the first positional arg as the ``event`` key.
            _log.error(
                "RUN_FAILED",
                run_id=run_id,
                error=err_msg,
                exc_info=exc,
            )
            try:
                await writer.finish_run(run_id, status=RunStatus.FAILED, error=err_msg)
            except Exception as write_exc:
                _log.error(
                    "failed to update run row after RUN_FAILED",
                    run_id=run_id,
                    exc_info=write_exc,
                )
            _emit_posthog(
                posthog_client,
                "benchmark_run_failed",
                {
                    "benchmark_kind": benchmark_kind,
                    "smoke": smoke,
                    "$process_person_profile": False,
                },
            )
            raise

        finally:
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.remove_signal_handler(signal.SIGTERM)
            with contextlib.suppress(Exception):
                await _close_http_clients()
            if posthog_client is not None:
                with contextlib.suppress(Exception):
                    posthog_client.shutdown()  # type: ignore[no-untyped-call]
                atexit.unregister(posthog_client.shutdown)
