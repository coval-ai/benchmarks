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
  without triggering import errors.
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
import contextlib
import hashlib
import importlib
import signal
from datetime import UTC, datetime  # noqa: UP017 — UTC alias requires 3.11+, target is 3.12
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel

from coval_bench.runner.config import DEFAULT_STT_MATRIX, DEFAULT_TTS_MATRIX, ProviderEntry
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
    entry: ProviderEntry,
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

        try:
            async with asyncio.timeout(_STT_TIMEOUT_S):
                transcription_result = await with_retry(
                    lambda: provider.measure_ttft(
                        audio_bytes,
                        1,  # mono
                        2,  # PCM_16 = 2 bytes/sample
                        16000,
                        0.1,
                        duration_sec,
                    ),
                )
        except Exception as exc:
            err = _truncate(str(exc))
            _log.warning(
                "STT provider call failed",
                provider=entry.provider,
                model=entry.model,
                audio=str(audio_path),
                exc_info=exc,
            )
            results.append(
                Result(
                    run_id=run_id,
                    provider=entry.provider,
                    model=entry.model,
                    benchmark=Benchmark.STT,
                    metric_type="TTFT",
                    metric_value=None,
                    metric_units=None,
                    audio_filename=audio_path.name,
                    transcript=None,
                    status=ResultStatus.FAILED,
                    error=err,
                )
            )
            return results

        # 1. TTFT
        results.append(
            Result(
                run_id=run_id,
                provider=entry.provider,
                model=entry.model,
                benchmark=Benchmark.STT,
                metric_type="TTFT",
                metric_value=transcription_result.ttft_seconds,
                metric_units="seconds",
                audio_filename=audio_path.name,
                transcript=transcription_result.complete_transcript,
                status=ResultStatus.SUCCESS,
                error=None,
            )
        )

        # 2. AudioToFinal
        results.append(
            Result(
                run_id=run_id,
                provider=entry.provider,
                model=entry.model,
                benchmark=Benchmark.STT,
                metric_type="AudioToFinal",
                metric_value=transcription_result.audio_to_final_seconds,
                metric_units="seconds",
                audio_filename=audio_path.name,
                transcript=transcription_result.complete_transcript,
                status=ResultStatus.SUCCESS,
                error=None,
            )
        )

        # 3. RTF — suppress on invalid inputs (e.g. zero duration) rather than failing the row
        rtf_value: float | None = None
        if transcription_result.audio_to_final_seconds is not None and duration_sec > 0:
            with contextlib.suppress(Exception):
                rtf_value = compute_rtf(transcription_result.audio_to_final_seconds, duration_sec)

        results.append(
            Result(
                run_id=run_id,
                provider=entry.provider,
                model=entry.model,
                benchmark=Benchmark.STT,
                metric_type="RTF",
                metric_value=rtf_value,
                metric_units="ratio",
                audio_filename=audio_path.name,
                transcript=transcription_result.complete_transcript,
                status=ResultStatus.SUCCESS,
                error=None,
            )
        )

        # 4. WER (when ground-truth available)
        if transcript_ref and transcription_result.complete_transcript is not None:
            try:
                wer_result = compute_wer(transcript_ref, transcription_result.complete_transcript)
                results.append(
                    Result(
                        run_id=run_id,
                        provider=entry.provider,
                        model=entry.model,
                        benchmark=Benchmark.STT,
                        metric_type="WER",
                        metric_value=wer_result.wer_percentage,
                        metric_units="percent",
                        audio_filename=audio_path.name,
                        transcript=transcription_result.complete_transcript,
                        status=ResultStatus.SUCCESS,
                        error=None,
                    )
                )
            except Exception as exc:
                _log.warning(
                    "WER computation failed",
                    provider=entry.provider,
                    model=entry.model,
                    exc_info=exc,
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
    entry: ProviderEntry,
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

        try:
            async with asyncio.timeout(_TTS_TIMEOUT_S):
                tts_result = await with_retry(
                    lambda: provider.synthesize(transcript),
                )
            audio_path = tts_result.audio_path
        except Exception as exc:
            err = _truncate(str(exc))
            _log.warning(
                "TTS provider call failed",
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
                    metric_type="TTFA",
                    metric_value=None,
                    metric_units=None,
                    audio_filename=None,
                    transcript=None,
                    status=ResultStatus.FAILED,
                    error=err,
                )
            )
            return results

        try:
            # 1. TTFA
            results.append(
                Result(
                    run_id=run_id,
                    provider=entry.provider,
                    model=entry.model,
                    voice=entry.voice,
                    benchmark=Benchmark.TTS,
                    metric_type="TTFA",
                    metric_value=tts_result.ttfa_ms,
                    metric_units="milliseconds",
                    audio_filename=audio_path.name if audio_path else None,
                    transcript=transcript,
                    status=ResultStatus.SUCCESS,
                    error=None,
                )
            )

            # 2. WER via Whisper transcription of synthesized audio
            if audio_path is not None and audio_path.exists():
                try:
                    whisper_transcript = await _transcribe_with_whisper(audio_path, settings)
                    wer_result = compute_wer(transcript, whisper_transcript)
                    results.append(
                        Result(
                            run_id=run_id,
                            provider=entry.provider,
                            model=entry.model,
                            voice=entry.voice,
                            benchmark=Benchmark.TTS,
                            metric_type="WER",
                            metric_value=wer_result.wer_percentage,
                            metric_units="percent",
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


async def run_benchmarks(
    *,
    settings: Settings,
    benchmark_kind: Literal["stt", "tts", "both"] = "both",
    smoke: bool = False,
    matrix_overrides: list[ProviderEntry] | None = None,
) -> RunSummary:
    """Execute one complete benchmark run.

    Args:
        settings: Injected application settings (no global state).
        benchmark_kind: Which benchmark(s) to run.
        smoke: If True, process only the first dataset item (local dev mode).
        matrix_overrides: Optional list of ``ProviderEntry`` objects that override
            the default matrices by ``(provider, model)`` key.

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

    # ------------------------------------------------------------------
    # 1. Resolve + filter provider matrix
    # ------------------------------------------------------------------
    stt_matrix = list(DEFAULT_STT_MATRIX)
    tts_matrix = list(DEFAULT_TTS_MATRIX)

    if matrix_overrides:
        override_map: dict[tuple[str, str], ProviderEntry] = {
            (e.provider, e.model): e for e in matrix_overrides
        }
        stt_matrix = [override_map.get((e.provider, e.model), e) for e in stt_matrix]
        tts_matrix = [override_map.get((e.provider, e.model), e) for e in tts_matrix]

        # Also apply any overrides that add *new* (provider, model) pairs
        existing_stt_keys = {(e.provider, e.model) for e in stt_matrix}
        existing_tts_keys = {(e.provider, e.model) for e in tts_matrix}
        for ov in matrix_overrides:
            if (ov.provider, ov.model) not in existing_stt_keys and ov.voice is None:
                stt_matrix.append(ov)
            if (ov.provider, ov.model) not in existing_tts_keys and ov.voice is not None:
                tts_matrix.append(ov)

    enabled_stt = [e for e in stt_matrix if e.enabled and not e.disabled]
    enabled_tts = [e for e in tts_matrix if e.enabled and not e.disabled]

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

        run = await writer.start_run(
            runner_sha=settings.runner_sha,
            dataset_id=settings.dataset_id,
            dataset_sha256=dataset_sha256,
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

            _log.info(
                "benchmark run finished",
                run_id=run_id,
                status=str(final_status),
                total_results=total_results,
                success_count=success_count,
                fail_count=fail_count,
                duration_s=(finished_at - started_at).total_seconds(),
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
            _log.warning(
                "benchmark run finished early due to SIGTERM",
                run_id=run_id,
                status=str(RunStatus.PARTIAL),
                total_results=total_results,
                success_count=success_count,
                fail_count=fail_count,
                duration_s=(finished_at - started_at).total_seconds(),
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
            raise

        finally:
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.remove_signal_handler(signal.SIGTERM)
