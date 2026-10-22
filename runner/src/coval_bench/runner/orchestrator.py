# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN401 -- lazy runtime collaborators intentionally use Any in this module.

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
- Concurrency: a :class:`ModelGate` holds one lock per (provider, model) so a
  model never serves two of our requests at once, under a global cap of 8 open
  provider connections. Different models run side by side.
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
import importlib
import random
import signal
import wave
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime  # noqa: UP017 — UTC alias requires 3.11+, target is 3.12
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog
from posthog import Posthog
from pydantic import BaseModel, Field

from coval_bench import telemetry
from coval_bench.datasets.suite import (
    DEDICATED_STT_SUITE,
    DEDICATED_TTS_SUITE,
    DEFAULT_STT_DATASET,
    SHARED_TTS_SUITE,
    stt_sample_size,
    tts_sample_size,
)
from coval_bench.db.registry_store import fetch_models
from coval_bench.logging import log_run_failed, log_run_partial
from coval_bench.providers._http_session import close_all as _close_http_clients
from coval_bench.providers.base import Provider
from coval_bench.registries import (
    METRIC_EXCLUSIONS,
    METRIC_SPECS,
    Benchmark,
    Metric,
    RegisteredModel,
    Source,
)
from coval_bench.runner.gate import ModelGate
from coval_bench.runner.retry import with_retry

if TYPE_CHECKING:
    from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.runner")

_CONCURRENCY_CAP = 8
_STT_TIMEOUT_S = 45
_TTS_TIMEOUT_S = 60
# Stable contract — matched by the reason classifier and the alerting log metric.
# Mirrors _SILENT_FAILURE_PREFIX in providers/tts/_common.py.
_STT_SILENT_FAILURE = "provider closed the stream without sending a transcript or an error"

_MAX_ERROR_LEN = 4000  # truncate error messages stored in DB (Postgres text is unbounded
# but huge stack traces choke log pipelines)

# Items a fully-failed provider must have run before RUN_PARTIAL calls it dead;
_MIN_DEAD_ITEMS = 3


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class ProviderReliability(BaseModel):
    """Item outcomes for one ``(benchmark, provider, model)`` in a run.

    The three counts partition the items attempted. ``key_failures`` are failures
    the classifier attributes to our key (auth, credit, rate limit) rather than to
    the provider, so a provider's reliability is
    ``succeeded / (succeeded + provider_failures)``.
    """

    benchmark: str
    provider: str
    model: str
    succeeded: int
    provider_failures: int
    key_failures: int


class RunSummary(BaseModel):
    """Typed return value from ``run_benchmarks``."""

    run_id: int
    started_at: datetime
    finished_at: datetime
    status: str  # RunStatus StrEnum value — avoids hard import at module load
    total_results: int
    success_count: int
    fail_count: int
    reliability: list[ProviderReliability] = Field(default_factory=list)
    sigterm: bool = False


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
    *,
    missing_reason: str | None = None,
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
        return result_status.FAILED, missing_reason or f"no {metric_label} produced"
    return result_status.SUCCESS, None


def _stt_silent_failure(result: Any) -> str | None:  # noqa: ANN401 — TranscriptionResult
    """Diagnostic when a provider returned nothing at all and reported no reason.

    The STT mirror of the TTS guard in ``providers/tts/_common.py``: every streaming
    provider reaches its return by falling out of the receive loop, so an error frame
    whose schema the integration failed to match leaves ``error`` unset and the row
    lands on a generic "no <METRIC> produced".

    Deliberately narrower than the TTS guard. A provider can deliver a final segment
    with empty text, which still yields a legitimate TTFT/AudioToFinal measurement —
    failing the item on an empty transcript alone would discard real latency data and
    shrink sample counts. So this fires only when nothing whatsoever was produced.
    """
    if result.complete_transcript or result.partial_transcripts:
        return None
    if result.ttft_seconds is not None or result.audio_to_final_seconds is not None:
        return None
    return _STT_SILENT_FAILURE


_FINALIZATION_DEFAULTS: dict[str, Any] = {
    "finalization_latency_seconds": None,
    "finalization_trigger": None,
    "final_audio_window_end_seconds": None,
    "finalization_warning_code": None,
    "finalization_timed_out": False,
}


def _finalization_events(result: Any) -> dict[str, Any]:  # noqa: ANN401 — TranscriptionResult
    if result is None:
        return dict(_FINALIZATION_DEFAULTS)
    return {name: getattr(result, name) for name in _FINALIZATION_DEFAULTS}


def _final_missing_reason(result: Any) -> str | None:  # noqa: ANN401 — TranscriptionResult
    if result is None or result.audio_to_final_seconds is not None:
        return None
    if result.finalization_warning_code:
        return f"finalization ignored: {result.finalization_warning_code}"
    if result.finalization_timed_out:
        return "finalization timed out"
    return None


def _log_item_failures(
    event: str,
    results: list[Any],
    logged_reasons: set[str],
    result_status: Any,  # noqa: ANN401 — ResultStatus enum, lazy-imported by callers
    *,
    provider: str,
    model: str,
    item: str | None,
) -> None:
    """Emit one per-item warning for FAILED rows nothing else logged.

    A failed item fans out to several FAILED rows, so this logs once, keyed by
    metric, not per row. *logged_reasons* carries messages already warned at
    their source (exceptions, metric crashes) so they are not logged twice.
    """
    reasons = {
        r.metric_type: r.error
        for r in results
        if r.status is result_status.FAILED
        and r.error is not None
        and r.error not in logged_reasons
    }
    if reasons:
        logger.warning(event, provider=provider, model=model, item=item, reasons=reasons)


def _record_item_metric(
    entry: RegisteredModel,
    results: list[Any],
    result_status: Any,  # noqa: ANN401 — ResultStatus enum, lazy-imported by callers
) -> None:
    """Count the item on ``benchmark_items``; a provider that produced no rows is not counted."""
    result = telemetry.item_result(results, result_status)
    if result is not None:
        telemetry.record_item(
            kind=str(entry.benchmark), provider=entry.provider, model=entry.model, result=result
        )


def _dead_providers(
    results: list[Any],
    result_status: Any,  # noqa: ANN401 — ResultStatus enum, lazy-imported by callers
) -> list[str]:
    """``benchmark:provider/model (reason)`` for each provider whose every row failed.

    Keyed by benchmark as well as provider and model, because a provider may be
    registered under more than one modality; keying on the provider alone would let a
    dead model be masked by that provider's healthy rows in the other benchmark. The
    reason is the most common error across the rows, so the alert names a cause.

    A provider only counts as dead when it failed at least ``_MIN_DEAD_ITEMS`` items.
    """
    rows: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for r in results:
        rows[(str(r.benchmark), r.provider, r.model)].append(r)

    dead: list[str] = []
    for (benchmark, provider, model), group in sorted(rows.items()):
        if not all(row.status == result_status.FAILED for row in group):
            continue
        items_run = max(Counter(row.metric_type for row in group).values())
        if items_run < _MIN_DEAD_ITEMS:
            continue
        ranked = Counter(row.error for row in group if row.error).most_common(1)
        reason = ranked[0][0] if ranked else "no reason reported"
        dead.append(f"{benchmark}:{provider}/{model} ({_truncate(reason)})")
    return dead


# Per benchmark, the one metric written for every item, never excluded for a
# provider, and failed only when the provider call failed. Its rows are the
# per-item verdict.
_ITEM_VERDICT_METRIC: dict[str, str] = {
    str(Benchmark.TTS): str(Metric.TTFA),
    str(Benchmark.STT): str(Metric.AUDIO_TO_FINAL),
}


def _provider_reliability(
    results: list[Any],
    result_status: Any,  # noqa: ANN401 — ResultStatus enum, lazy-imported by callers
) -> list[ProviderReliability]:
    """Item outcomes per ``(benchmark, provider, model)``, from the verdict-metric rows.

    A failed item counts against the provider unless its error classifies as a key
    problem on our side (auth, credit, rate limit), in which case it is reported
    separately so an exhausted key does not read as an unreliable provider.
    Benchmarks without a verdict metric are skipped.
    """
    from coval_bench.arena.provider_health import classify_failure

    rows: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for r in results:
        if str(r.metric_type) != _ITEM_VERDICT_METRIC.get(str(r.benchmark)):
            continue
        rows[(str(r.benchmark), r.provider, r.model)].append(r)

    out: list[ProviderReliability] = []
    for (benchmark, provider, model), group in sorted(rows.items()):
        succeeded = provider_failures = key_failures = 0
        for row in group:
            if row.status == result_status.SUCCESS:
                succeeded += 1
            elif classify_failure(None, row.error) is not None:
                key_failures += 1
            else:
                provider_failures += 1
        out.append(
            ProviderReliability(
                benchmark=benchmark,
                provider=provider,
                model=model,
                succeeded=succeeded,
                provider_failures=provider_failures,
                key_failures=key_failures,
            )
        )
    return out


def _log_run_outcome(
    final_status: Any,  # noqa: ANN401 — RunStatus enum, lazy-imported by callers
    results: list[Any],
    fail_count: int,
    *,
    result_status: Any,  # noqa: ANN401 — ResultStatus enum, lazy-imported by callers
    run_status: Any,  # noqa: ANN401 — RunStatus enum, lazy-imported by callers
) -> None:
    """Emit the run-level event that the infra alert metrics filter on.

    Storing the status on the run row is not enough to notify anyone: the Cloud
    Logging metrics in benchmark-infra match on the literal ``RUN_FAILED`` /
    ``RUN_PARTIAL`` event strings, so a run that ends badly and logs nothing is
    invisible in Slack.

    Deliberately log-only — it never raises. The runner job sets ``max_retries = 1``,
    so exiting non-zero would re-run the whole benchmark, re-spending on every
    provider and writing a duplicate run. Alerting keys off the log event, not the
    exit code, so the page fires either way.

    A total failure reports only ``RUN_FAILED``; the per-provider breakdown is
    skipped there because every provider is in it, and one incident must not page
    twice.

    A partial run reports ``RUN_PARTIAL`` **only when some provider failed every one
    of at least ``_MIN_DEAD_ITEMS`` items**, not on every partial run."""
    if final_status is run_status.FAILED:
        log_run_failed(
            f"all {fail_count} result rows failed"
            if results
            else "benchmark produced no results (no providers configured or dataset empty)"
        )
        return
    if final_status is not run_status.PARTIAL:
        return
    dead = _dead_providers(results, result_status)
    if dead:
        log_run_partial(f"providers failed every item: {'; '.join(dead)}")


async def _transcribe_with_whisper(audio_path: Path, settings: Settings) -> str:
    """Transcribe *audio_path* with OpenAI ``whisper-1`` and return the text.

    Used by the TTS path to compute WER against the synthesized audio.
    Retries once on OpenAI's transient gateway 404s (the SDK retries 429/5xx
    itself but not 404); otherwise raises on API error — caller handles and
    records as a failed Result.
    """
    import openai

    api_key = settings.openai_api_key
    if api_key is None:
        raise RuntimeError("openai_api_key is required for TTS-WER computation")

    client = openai.AsyncOpenAI(api_key=api_key.get_secret_value())

    async def attempt() -> str:
        with audio_path.open("rb") as fh:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=fh,
            )
        return str(response.text)

    return await with_retry(attempt, max_attempts=2, retry_on=(openai.NotFoundError,))


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


def _get_manifest_sha256() -> Any:  # noqa: ANN401
    """Resolve ``packaged_manifest_sha256`` at call time (lazy import)."""
    mod = importlib.import_module("coval_bench.datasets.loader")
    return mod.packaged_manifest_sha256


def _get_family_rng() -> Any:  # noqa: ANN401
    """Resolve ``family_rng`` at call time (lazy import)."""
    mod = importlib.import_module("coval_bench.datasets")
    return mod.family_rng


def _get_db_symbols() -> tuple[Any, Any, Any, Any]:
    """Return (lifespan_pool, RunWriter, RunStatus, db.models) at call time."""
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


@dataclass
class _ItemRows:
    """Result rows for one provider × item, built with the same identity fields every time."""

    entry: RegisteredModel
    run_id: int
    benchmark: Any
    result_cls: Any
    result_status: Any
    voice: str | None = None
    results: list[Any] = field(default_factory=list)
    # Reasons already warned at their source; the per-item summary skips these.
    logged_reasons: set[str] = field(default_factory=set)

    def add(
        self,
        metric: Metric,
        value: float | None,
        *,
        status: Any,
        error: str | None = None,
        audio_filename: str | None = None,
        transcript: str | None = None,
        units: bool = True,
        **extra: Any,
    ) -> None:
        self.results.append(
            self.result_cls(
                run_id=self.run_id,
                provider=self.entry.provider,
                model=self.entry.model,
                voice=self.voice,
                benchmark=self.benchmark,
                metric_type=metric,
                metric_value=value,
                metric_units=METRIC_SPECS[metric].units if units else None,
                audio_filename=audio_filename,
                transcript=transcript,
                status=status,
                error=error,
                **extra,
            )
        )

    def add_measured(
        self,
        metric: Metric,
        value: float | None,
        item_error: str | None,
        *,
        status_from: float | None = None,
        missing_reason: str | None = None,
        audio_filename: str | None,
        transcript: str | None,
        log_event: str,
        log_key: str,
    ) -> Any:
        """Add a latency row whose status follows ``_metric_outcome``; returns the status.

        ``status_from`` lets a derived metric (RTF) take its outcome from the
        measurement it derives from instead of its own value.
        """
        status, error = _metric_outcome(
            value if status_from is None else status_from,
            item_error,
            metric,
            self.result_status,
            missing_reason=missing_reason,
        )
        self.add(
            metric,
            value,
            status=status,
            error=error,
            audio_filename=audio_filename,
            transcript=transcript,
        )
        if status is self.result_status.SUCCESS and value is not None:
            logger.debug(
                log_event,
                provider=self.entry.provider,
                model=self.entry.model,
                **{log_key: value},
            )
        return status

    def score_wer(
        self,
        compute_wer: Any,
        reference: str,
        hypothesis: str,
        *,
        audio_filename: str | None,
        ok_event: str,
        failed_event: str,
    ) -> None:
        """One WER row. compute_wer is deterministic, so a crash is a real FAILED row."""
        try:
            wer_result = compute_wer(reference, hypothesis)
        except Exception as exc:
            self.logged_reasons.add(_truncate(str(exc)))
            logger.warning(
                failed_event, provider=self.entry.provider, model=self.entry.model, exc_info=exc
            )
            self.add(
                Metric.WER,
                None,
                status=self.result_status.FAILED,
                error=_truncate(str(exc)),
                audio_filename=audio_filename,
                transcript=hypothesis,
                units=False,
            )
            return
        self.add(
            Metric.WER,
            wer_result.wer_percentage,
            status=self.result_status.SUCCESS,
            audio_filename=audio_filename,
            transcript=hypothesis,
            **wer_result.error_percentages,
            **wer_result.error_counts,
        )
        logger.debug(
            ok_event,
            provider=self.entry.provider,
            model=self.entry.model,
            wer_percentage=wer_result.wer_percentage,
        )

    async def call_provider(
        self,
        kind: Literal["stt", "tts"],
        timeout_s: float,
        make_provider: Callable[[], Any],
        call: Callable[[Any], Awaitable[Any]],
        **log_fields: Any,
    ) -> tuple[Any, str | None]:
        """Build the provider and call it with timeout + retry. Returns (result, item_error).

        Construction sits inside the try so a config error (an unset endpoint URL,
        say) lands as error rows instead of vanishing into the gather.
        """
        try:
            provider = make_provider()
            async with asyncio.timeout(timeout_s):
                result = await with_retry(lambda: call(provider))
        except Exception as exc:
            item_error = _truncate(str(exc))
            self.logged_reasons.add(item_error)
            logger.warning(
                f"{kind}_provider_call_failed",
                provider=self.entry.provider,
                model=self.entry.model,
                exc_info=exc,
                **log_fields,
            )
            return None, item_error
        return result, None

    def finish(self, kind: Literal["stt", "tts"], item: str | None) -> list[Any]:
        _log_item_failures(
            f"{kind}_item_failed",
            self.results,
            self.logged_reasons,
            self.result_status,
            provider=self.entry.provider,
            model=self.entry.model,
            item=item,
        )
        _record_item_metric(self.entry, self.results, self.result_status)
        return self.results


def _item_rows(
    entry: RegisteredModel, run_id: int, kind: Literal["stt", "tts"], voice: str | None = None
) -> _ItemRows:
    _, _, _, models_mod = _get_db_symbols()
    return _ItemRows(
        entry=entry,
        run_id=run_id,
        benchmark=models_mod.Benchmark(kind.upper()),
        result_cls=models_mod.Result,
        result_status=models_mod.ResultStatus,
        voice=voice,
    )


async def _persist_item(
    *,
    writer: Any,
    artifact_client: Any,
    settings: Settings,
    capture_outcomes: list[Any] | None,
    **envelope: Any,
) -> Any:
    from coval_bench.runner.persistence import persist_capture, prepare_capture_envelope

    outcome = await persist_capture(
        writer=writer,
        storage_client=artifact_client,
        bucket=settings.benchmark_artifact_bucket,
        envelope=prepare_capture_envelope(**envelope),
    )
    if capture_outcomes is not None:
        capture_outcomes.append(outcome)
    return outcome


def _read_pcm16(audio_path: Path) -> bytes:
    """Frames only: a WAV header sent as PCM reads as a click that can derail provider VAD."""
    with wave.open(str(audio_path), "rb") as wav_file:
        fmt = (wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth())
        if fmt != (16000, 1, 2):
            raise ValueError(f"{audio_path.name}: expected 16 kHz mono PCM16, got {fmt}")
        return wav_file.readframes(wav_file.getnframes())


def _trim_to_speech_end(
    audio_bytes: bytes, duration_sec: float, speech_end_offset_ms: object
) -> tuple[bytes, float]:
    """Cut trailing silence after the VAD speech end; the duration follows what was sent."""
    if not isinstance(speech_end_offset_ms, (int, float)):
        return audio_bytes, duration_sec
    trailing_ms = max(0.0, duration_sec * 1000.0 - speech_end_offset_ms)
    tail_bytes = int(round(trailing_ms / 1000.0 * 16000)) * 2
    if 0 < tail_bytes < len(audio_bytes):
        return audio_bytes[: len(audio_bytes) - tail_bytes], speech_end_offset_ms / 1000.0
    return audio_bytes, duration_sec


async def _run_stt_item(
    *,
    entry: RegisteredModel,
    item: Any,  # noqa: ANN401 — DatasetItem is a runtime-typed sibling-agent type
    run_id: int,
    gate: ModelGate,
    settings: Settings,
    writer: Any | None = None,  # noqa: ANN401 — RunWriter, lazy-imported in caller
    dataset_id: str | None = None,
    dataset_sha256: str = "",
    artifact_client: Any | None = None,
    capture_outcomes: list[Any] | None = None,
) -> list[Any]:
    """One STT provider × item: TTFT, AudioToFinal, TTFS, RTF, WER rows.

    With a *writer* the rows are persisted before returning, so a task that
    completes is stored even if the job is cancelled later.
    """
    stt_providers = _get_stt_providers()
    compute_wer, compute_rtf = _get_metrics()
    if writer is not None and artifact_client is None:
        raise RuntimeError("required STT capture is not initialized")
    rows = _item_rows(entry, run_id, "stt")

    async with gate.slot((entry.provider, entry.model)):
        provider_cls = stt_providers.get(entry.provider)
        if provider_cls is None:
            logger.warning("unknown_stt_provider", provider=entry.provider, model=entry.model)
            return []

        audio_path: Path = item.path
        transcript_ref: str = item.transcript
        speech_end_offset_ms = item.speech_end_offset_ms
        audio_bytes, duration_sec = _trim_to_speech_end(
            _read_pcm16(audio_path), item.duration_sec, speech_end_offset_ms
        )

        transcription_result, item_error = await rows.call_provider(
            "stt",
            _STT_TIMEOUT_S,
            lambda: provider_cls(settings, entry.model),
            # mono, PCM_16 = 2 bytes/sample, 16 kHz
            lambda provider: provider.measure_ttft(audio_bytes, 1, 2, 16000, 0.1),
            audio=str(audio_path),
        )
        item_error = item_error or (
            transcription_result.error if transcription_result is not None else None
        )
        if transcription_result is not None and item_error is None:
            item_error = _stt_silent_failure(transcription_result)
        ttft_seconds = transcription_result.ttft_seconds if transcription_result else None
        audio_to_final = (
            transcription_result.audio_to_final_seconds if transcription_result else None
        )
        complete_transcript = (
            transcription_result.complete_transcript if transcription_result else None
        )
        final_missing_reason = _final_missing_reason(transcription_result)
        excluded = {
            m
            for m in (Metric.TTFT, Metric.TTFS)
            if (entry.provider, entry.model) in METRIC_EXCLUSIONS[m]
        }

        if Metric.TTFT not in excluded:
            rows.add_measured(
                Metric.TTFT,
                ttft_seconds,
                item_error,
                log_event="stt_ttft",
                log_key="ttft_seconds",
                audio_filename=audio_path.name,
                transcript=complete_transcript,
            )
        rows.add_measured(
            Metric.AUDIO_TO_FINAL,
            audio_to_final,
            item_error,
            missing_reason=final_missing_reason,
            log_event="stt_audio_to_final",
            log_key="audio_to_final_seconds",
            audio_filename=audio_path.name,
            transcript=complete_transcript,
        )

        # TTFS: time-to-final from VAD end-of-speech, the headline metric. A missing offset
        # fails the row; an early final (ahead of the anchor) clamps to 0 in compute_ttfs.
        ttfs_value: float | None = None
        ttfs_calc_error: str | None = None
        if audio_to_final is not None and isinstance(speech_end_offset_ms, (int, float)):
            try:
                compute_ttfs = importlib.import_module("coval_bench.metrics").compute_ttfs
                ttfs_value = compute_ttfs(audio_to_final, speech_end_offset_ms / 1000.0)
            except Exception as exc:
                ttfs_calc_error = str(exc)
                rows.logged_reasons.add(_truncate(ttfs_calc_error))
                logger.warning(
                    "ttfs_computation_failed",
                    provider=entry.provider,
                    model=entry.model,
                    exc_info=exc,
                )
        if Metric.TTFS not in excluded:
            rows.add_measured(
                Metric.TTFS,
                ttfs_value,
                item_error or ttfs_calc_error,
                missing_reason=final_missing_reason,
                log_event="stt_ttfs",
                log_key="ttfs_seconds",
                audio_filename=audio_path.name,
                transcript=complete_transcript,
            )

        # RTF derives from AudioToFinal, so its status tracks whether a final arrived. A
        # present final with an uncomputable RTF (zero duration) stays a null-valued success.
        rtf_value: float | None = None
        if audio_to_final is not None and duration_sec > 0:
            try:
                rtf_value = compute_rtf(audio_to_final, duration_sec)
            except Exception as exc:
                logger.warning(
                    "rtf_computation_failed",
                    provider=entry.provider,
                    model=entry.model,
                    exc_info=exc,
                )
        rows.add_measured(
            Metric.RTF,
            rtf_value,
            item_error,
            status_from=audio_to_final,
            missing_reason=final_missing_reason,
            log_event="stt_rtf",
            log_key="rtf",
            audio_filename=audio_path.name,
            transcript=complete_transcript,
        )

        # WER only with ground truth and a clean stream: never score a transcript
        # salvaged from a failed run.
        if transcript_ref and complete_transcript is not None and item_error is None:
            rows.score_wer(
                compute_wer,
                transcript_ref,
                complete_transcript,
                audio_filename=audio_path.name,
                ok_event="stt_wer",
                failed_event="wer_computation_failed",
            )

    results = rows.finish("stt", audio_path.name)
    if writer is not None and artifact_client is not None:
        await _persist_item(
            writer=writer,
            artifact_client=artifact_client,
            settings=settings,
            capture_outcomes=capture_outcomes,
            run_id=run_id,
            dataset_id=dataset_id or settings.dataset_id or DEFAULT_STT_DATASET,
            dataset_sha256=dataset_sha256,
            sample_id=item.sample_id or audio_path.name,
            entry=entry,
            benchmark=rows.benchmark,
            results=results,
            provider_error=item_error,
            captured_at=datetime.now(UTC),
            transcript=complete_transcript,
            timing_events={
                "ttft_seconds": ttft_seconds,
                "audio_to_final_seconds": audio_to_final,
                "speech_end_offset_ms": speech_end_offset_ms,
                "effective_duration_sec": duration_sec,
                **_finalization_events(transcription_result),
            },
        )
    return results


# ---------------------------------------------------------------------------
# TTS coroutine builder
# ---------------------------------------------------------------------------


def _assign_tts_voices(entry: RegisteredModel, item_count: int, run_id: int) -> list[str | None]:
    """Assign a voice to each dataset item, one entry per item in dataset order.

    Models with a ``voices`` pool get an exactly balanced split (counts differ
    by at most one when the pool doesn't divide the item count evenly). Which
    item lands on which voice is shuffled with a run/model-derived seed, so the
    text↔voice pairing rotates across runs while staying reproducible for a
    given run.
    """
    if not entry.voices:
        return [entry.voice] * item_count
    assignments: list[str | None] = [
        entry.voices[i % len(entry.voices)].id for i in range(item_count)
    ]
    rng = random.Random(f"{run_id}:{entry.provider}:{entry.model}")  # noqa: S311
    rng.shuffle(assignments)
    return assignments


def _ttfa_transport_gate(tts_result: Any) -> str | None:
    """Why a TTFA is not comparable to the warm HTTP/2 cohort, or None when it is."""
    if tts_result.http_version is not None and tts_result.http_version != "HTTP/2":
        return _truncate(
            f"TTFA measured over {tts_result.http_version}; not comparable "
            "(no HTTP/2 multiplexing, TTFA reabsorbs TCP+TLS)"
        )
    if tts_result.connection_reused is False:
        return _truncate(
            "TTFA measured over a cold connection; not comparable (TTFA reabsorbs TCP+TLS)"
        )
    return None


async def _run_tts_item(
    *,
    entry: RegisteredModel,
    item: Any,  # noqa: ANN401 — TTSDatasetItem is a runtime-typed sibling-agent type
    run_id: int,
    gate: ModelGate,
    settings: Settings,
    voice: str | None = None,
    writer: Any | None = None,  # noqa: ANN401 — RunWriter, lazy-imported in caller
    dataset_id: str = "tts-v1",
    dataset_sha256: str = "",
    artifact_client: Any | None = None,
    capture_outcomes: list[Any] | None = None,
) -> list[Any]:
    """One TTS provider × item: TTFA (plus its components) and Whisper-scored WER rows.

    *voice* overrides the entry's pinned voice for the balanced split. With a
    *writer* the rows are persisted before returning, as for STT.
    """
    voice = voice if voice is not None else entry.voice
    tts_providers = _get_tts_providers()
    compute_wer, _ = _get_metrics()
    if writer is not None and artifact_client is None:
        raise RuntimeError("required TTS capture is not initialized")
    rows = _item_rows(entry, run_id, "tts", voice)
    ResultStatus = rows.result_status
    audio_path: Path | None = None
    capture_outcome: Any | None = None

    async with gate.slot((entry.provider, entry.model)):
        provider_cls = tts_providers.get(entry.provider)
        if provider_cls is None:
            logger.warning("unknown_tts_provider", provider=entry.provider, model=entry.model)
            return []

        transcript: str = item.transcript
        tts_result, item_error = await rows.call_provider(
            "tts",
            _TTS_TIMEOUT_S,
            lambda: provider_cls(settings=settings, model=entry.model, voice=voice),
            lambda provider: provider.synthesize(transcript),
        )
        if tts_result is not None:
            audio_path = tts_result.audio_path
        # One failure model: a raised exception or a provider-reported error fails the
        # TTFA row and suppresses WER scoring.
        item_error = item_error or (tts_result.error if tts_result is not None else None)
        ttfa_ms = tts_result.ttfa_ms if tts_result else None

        try:
            ttfa_status, ttfa_error = _metric_outcome(
                ttfa_ms, item_error, Metric.TTFA, ResultStatus
            )
            ttfa_value = ttfa_ms
            # A measurement over HTTP/1.1 or a cold socket becomes a null-valued success:
            # excluded from aggregation, not a failure. Only a would-be SUCCESS is overridden.
            if ttfa_status is ResultStatus.SUCCESS and tts_result is not None:
                gate_reason = _ttfa_transport_gate(tts_result)
                if gate_reason is not None:
                    ttfa_error, ttfa_value = gate_reason, None
            rows.add(
                Metric.TTFA,
                ttfa_value,
                status=ttfa_status,
                error=ttfa_error,
                audio_filename=audio_path.name if audio_path else None,
                transcript=transcript,
            )

            # Component rows only for a successful TTFA with a known split, so the two
            # always sum back to the TTFA row. A failed, nulled, or arrival-only TTFA
            # writes neither.
            leading_silence_ms = tts_result.leading_silence_ms if tts_result else None
            if (
                ttfa_status is ResultStatus.SUCCESS
                and ttfa_value is not None
                and leading_silence_ms is not None
            ):
                for component, value in (
                    (Metric.TTFA_ROUNDTRIP, ttfa_value - leading_silence_ms),
                    (Metric.TTFA_LEADING_SILENCE, leading_silence_ms),
                ):
                    rows.add(component, value, status=ResultStatus.SUCCESS, transcript=transcript)

            # WER via Whisper on the synthesized audio. Whisper is our instrument, not the
            # provider under test: a transcription failure is logged and skipped so a
            # Whisper outage doesn't drag the provider's run to PARTIAL.
            if item_error is None and audio_path is not None and audio_path.exists():
                whisper_transcript: str | None = None
                try:
                    whisper_transcript = await _transcribe_with_whisper(audio_path, settings)
                except Exception as exc:
                    logger.warning(
                        "tts_wer_transcription_failed",
                        provider=entry.provider,
                        model=entry.model,
                        exc_info=exc,
                    )
                if whisper_transcript is not None:
                    rows.score_wer(
                        compute_wer,
                        item.spoken_reference or transcript,
                        whisper_transcript,
                        audio_filename=audio_path.name,
                        ok_event="tts_wer",
                        failed_event="tts_wer_computation_failed",
                    )
            if writer is not None and artifact_client is not None:
                from coval_bench.observation_artifacts import snapshot_generated_audio

                capture_outcome = await _persist_item(
                    writer=writer,
                    artifact_client=artifact_client,
                    settings=settings,
                    capture_outcomes=capture_outcomes,
                    run_id=run_id,
                    dataset_id=dataset_id,
                    dataset_sha256=dataset_sha256,
                    sample_id=item.testcase_id,
                    entry=entry,
                    benchmark=rows.benchmark,
                    results=rows.results,
                    provider_error=item_error,
                    captured_at=datetime.now(UTC),
                    voice=voice,
                    timing_events={"ttfa_ms": ttfa_ms},
                    audio_snapshot=(
                        snapshot_generated_audio(audio_path) if audio_path is not None else None
                    ),
                    transport_protocol=tts_result.http_version if tts_result else None,
                    submit_to_headers_ms=tts_result.submit_to_headers_ms if tts_result else None,
                )
        finally:
            # This module owns audio cleanup, not the provider.
            safe_cleanup = str(capture_outcome) in {"completed", "pending"}
            if audio_path is not None and audio_path.exists() and (writer is None or safe_cleanup):
                try:
                    audio_path.unlink()
                except OSError as exc:
                    logger.warning(
                        "tts_audio_file_delete_failed", path=str(audio_path), exc_info=exc
                    )

    return rows.finish("tts", item.testcase_id)


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
        logger.warning("posthog_emit_failed", event_name=event, exc_info=True)


# Transient-failure retry for the end-of-run bucket refresh.
_BUCKET_REFRESH_ATTEMPTS = 3
_BUCKET_REFRESH_RETRY_DELAY_S = 0.5


async def _refresh_series_bucket(writer: Any, run_id: int) -> None:  # noqa: ANN401 — RunWriter, lazy-imported by the caller
    """Best-effort refresh of the rollup bucket."""
    for attempt in range(1, _BUCKET_REFRESH_ATTEMPTS + 1):
        try:
            await writer.rebuild_run_rollup(run_id)
            return
        except Exception:
            if attempt == _BUCKET_REFRESH_ATTEMPTS:
                logger.warning("normalized_series_bucket_refresh_failed", exc_info=True)
                return
            logger.info("normalized_series_bucket_refresh_retry", attempt=attempt)
            await asyncio.sleep(_BUCKET_REFRESH_RETRY_DELAY_S)


def _current_tick(settings: Settings) -> datetime:
    period = settings.schedule_period_seconds
    epoch = int(datetime.now(tz=UTC).timestamp())
    return datetime.fromtimestamp(epoch - epoch % period, tz=UTC)


def _runs_daily(entry: RegisteredModel) -> bool:
    """Dedicated endpoints run daily; so do Guava's shared ones, at Guava's request."""
    return entry.source is Source.DEDICATED_INFERENCE or entry.provider == "guava"


async def run_suite(
    *,
    settings: Settings,
    benchmark_kind: Literal["stt", "tts", "both"] = "both",
    smoke: bool = False,
    source: Literal["shared", "dedicated"] = "shared",
    only: frozenset[str] | None = None,
) -> list[RunSummary]:
    """Dedicated STT walks the suite, one run row per dataset on one tick; else a single run."""
    if source != "dedicated" or benchmark_kind != "stt" or settings.dataset_id is not None:
        return [
            await run_benchmarks(
                settings=settings,
                benchmark_kind=benchmark_kind,
                smoke=smoke,
                source=source,
                only=only,
            )
        ]

    scheduled_at = _current_tick(settings)
    summaries: list[RunSummary] = []
    for dataset_id in DEDICATED_STT_SUITE:
        try:
            summary = await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=smoke,
                source=source,
                dataset_id=dataset_id,
                scheduled_at=scheduled_at,
                only=only,
            )
        except Exception:
            logger.exception("suite_dataset_failed", dataset_id=dataset_id)
            continue
        summaries.append(summary)
        if summary.sigterm:
            break
    return summaries


@dataclass
class _RunContext:
    """What every phase of a run needs once the run row exists."""

    settings: Settings
    writer: Any
    run_id: int
    started_at: datetime
    artifact_client: Any
    posthog_client: Posthog | None
    benchmark_kind: Literal["stt", "tts", "both"]
    smoke: bool
    result_cls: Any
    result_status: Any
    run_status: Any
    capture_pending_error: str


@dataclass
class _RunTally:
    """Row counts for the summary, built the same way on every exit path."""

    typed_results: list[Any]
    success_count: int
    fail_count: int
    reliability: list[ProviderReliability]

    @property
    def total_results(self) -> int:
        return len(self.typed_results)


@dataclass(frozen=True)
class _DatasetPlan:
    stt_dataset_id: str
    stt_size: int | None
    tts_dataset_id: str
    tts_size: int | None


class _SigtermGuard:
    """Cloud Run sends SIGTERM ~10s before SIGKILL; cancel the main task so the run
    row lands as PARTIAL instead of sticking at 'running'."""

    def __init__(self) -> None:
        self.received = False
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()

    def install(self) -> None:
        with contextlib.suppress(NotImplementedError):
            self._loop.add_signal_handler(signal.SIGTERM, self._on_sigterm)

    def uninstall(self) -> None:
        with contextlib.suppress(NotImplementedError, ValueError):
            self._loop.remove_signal_handler(signal.SIGTERM)

    def uncancel(self) -> None:
        if self._task is not None:
            self._task.uncancel()

    def _on_sigterm(self) -> None:
        if self.received:
            return
        self.received = True
        logger.warning("sigterm_received")
        if self._task is not None:
            self._task.cancel()


def _init_posthog(settings: Settings) -> Posthog | None:
    if settings.posthog_disabled or not settings.posthog_project_token:
        return None
    try:
        client = Posthog(
            settings.posthog_project_token,
            host=settings.posthog_host,
            enable_exception_autocapture=True,
        )
    except Exception:
        logger.warning("posthog_init_failed", exc_info=True)
        return None
    atexit.register(client.shutdown)
    return client


def _shutdown_posthog(client: Posthog | None) -> None:
    if client is None:
        return
    with contextlib.suppress(Exception):
        client.shutdown()  # type: ignore[no-untyped-call]
    atexit.unregister(client.shutdown)


def _plan_datasets(
    settings: Settings,
    *,
    source: Literal["shared", "dedicated"],
    dataset_id: str | None,
    only: frozenset[str] | None,
) -> _DatasetPlan:
    stt_dataset_id = dataset_id or settings.dataset_id or DEFAULT_STT_DATASET
    tts_dataset_id = settings.tts_dataset_id
    if only and "tts_dataset_id" not in settings.model_fields_set:
        tts_dataset_id = next(
            iter(DEDICATED_TTS_SUITE if source == "dedicated" else SHARED_TTS_SUITE)
        )
    tts_override = (
        settings.tts_dataset_sample_size
        if settings.tts_dataset_sample_size is not None
        else settings.dataset_sample_size
    )
    return _DatasetPlan(
        stt_dataset_id=stt_dataset_id,
        stt_size=stt_sample_size(source, stt_dataset_id, settings.dataset_sample_size),
        tts_dataset_id=tts_dataset_id,
        tts_size=tts_sample_size(source, tts_dataset_id, tts_override),
    )


def _apply_matrix_overrides(
    models: list[RegisteredModel], overrides: list[RegisteredModel] | None
) -> tuple[list[RegisteredModel], list[RegisteredModel]]:
    """The registry split by benchmark, with overrides replacing or adding entries."""
    stt_matrix = [m for m in models if m.benchmark is Benchmark.STT]
    tts_matrix = [m for m in models if m.benchmark is Benchmark.TTS]
    if not overrides:
        return stt_matrix, tts_matrix
    override_map = {(e.benchmark, e.provider, e.model): e for e in overrides}
    stt_matrix = [override_map.get((e.benchmark, e.provider, e.model), e) for e in stt_matrix]
    tts_matrix = [override_map.get((e.benchmark, e.provider, e.model), e) for e in tts_matrix]
    existing_keys = {(e.benchmark, e.provider, e.model) for e in stt_matrix + tts_matrix}
    for ov in overrides:
        if (ov.benchmark, ov.provider, ov.model) not in existing_keys:
            (stt_matrix if ov.benchmark is Benchmark.STT else tts_matrix).append(ov)
    return stt_matrix, tts_matrix


def _select_models(
    stt_matrix: list[RegisteredModel],
    tts_matrix: list[RegisteredModel],
    *,
    source: Literal["shared", "dedicated"],
    benchmark_kind: Literal["stt", "tts", "both"],
    only: frozenset[str] | None,
) -> tuple[list[RegisteredModel], list[RegisteredModel]]:
    """The collected models this run owns. Published or not, they all run; only the
    API hides unpublished ones."""
    dedicated = source == "dedicated"
    enabled_stt = [e for e in stt_matrix if e.collected and _runs_daily(e) == dedicated]
    enabled_tts = [e for e in tts_matrix if e.collected and _runs_daily(e) == dedicated]
    if not only:
        return enabled_stt, enabled_tts
    enabled_stt = [e for e in enabled_stt if f"{e.provider}/{e.model}" in only]
    enabled_tts = [e for e in enabled_tts if f"{e.provider}/{e.model}" in only]
    in_run = (enabled_stt if benchmark_kind in ("stt", "both") else []) + (
        enabled_tts if benchmark_kind in ("tts", "both") else []
    )
    missing = only - {f"{e.provider}/{e.model}" for e in in_run}
    if missing:
        raise ValueError(
            f"--only matched no model collected for a {source} {benchmark_kind} run: "
            f"{sorted(missing)}"
        )
    return enabled_stt, enabled_tts


def _manifest_sha256(dataset_id: str, fallback: str) -> str:
    try:
        return str(_get_manifest_sha256()(dataset_id))
    except Exception:
        return fallback


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _capture_client() -> Any:
    try:
        from google.cloud import storage

        return storage.Client()
    except Exception as exc:
        raise RuntimeError("capture client initialization failed") from exc


def _require_dataset_hashes(
    benchmark_kind: Literal["stt", "tts", "both"], dataset_sha256: str, tts_manifest_sha256: str
) -> None:
    if benchmark_kind in ("stt", "both") and not _is_sha256(dataset_sha256):
        raise RuntimeError("required STT capture needs a valid dataset hash")
    if benchmark_kind in ("tts", "both") and len(tts_manifest_sha256) != 64:
        raise RuntimeError("required TTS capture needs a valid dataset hash")


def _stt_capture_ids(
    run_id: int, dataset_id: str, items: list[Any], entries: list[RegisteredModel]
) -> list[str]:
    from coval_bench.runner.capture import build_capture_identity, identity_digest

    return [
        identity_digest(
            build_capture_identity(
                run_id=run_id,
                benchmark="STT",
                dataset_id=dataset_id,
                sample_id=item.sample_id or item.path.name,
                provider=entry.provider,
                model=entry.model,
            )
        )
        for item in items
        for entry in entries
    ]


def _tts_capture_ids(
    run_id: int,
    dataset_id: str,
    items: list[Any],
    entries: list[RegisteredModel],
    voices: dict[tuple[str, str], list[str | None]],
) -> list[str]:
    from coval_bench.runner.capture import build_capture_identity, identity_digest

    return [
        identity_digest(
            build_capture_identity(
                run_id=run_id,
                benchmark="TTS",
                dataset_id=dataset_id,
                sample_id=item.testcase_id,
                provider=entry.provider,
                model=entry.model,
                voice=voices[(entry.provider, entry.model)][index],
            )
        )
        for index, item in enumerate(items)
        for entry in entries
    ]


async def _upload_manifest(
    ctx: _RunContext,
    *,
    scheduled_at: datetime,
    source: Literal["shared", "dedicated"],
    datasets: dict[str, str],
    expected_capture_ids: list[str],
) -> None:
    from coval_bench.runner.capture import RunManifest, upload_run_state

    manifest = RunManifest(
        run_id=ctx.run_id,
        scheduled_at=scheduled_at,
        benchmark_kind=ctx.benchmark_kind,
        source=source,
        datasets=datasets,
        expected_capture_ids=sorted(expected_capture_ids),
    )
    await asyncio.to_thread(
        upload_run_state,
        ctx.artifact_client,
        ctx.settings.benchmark_artifact_bucket,
        ctx.run_id,
        "manifest",
        manifest,
    )


async def _warm_providers(
    settings: Settings,
    benchmark_kind: Literal["stt", "tts", "both"],
    enabled_stt: list[RegisteredModel],
    enabled_tts: list[RegisteredModel],
) -> None:
    """Pay cold-start cost (TCP+TLS, gRPC channel) before t0. Warmup is a no-op unless
    a provider overrides it, and errors are non-fatal."""
    provider_classes: set[type[Provider]] = set()
    if benchmark_kind in ("stt", "both"):
        stt_providers = _get_stt_providers()
        provider_classes.update(
            cls for e in enabled_stt if (cls := stt_providers.get(e.provider)) is not None
        )
    if benchmark_kind in ("tts", "both"):
        tts_providers = _get_tts_providers()
        provider_classes.update(
            cls for e in enabled_tts if (cls := tts_providers.get(e.provider)) is not None
        )
    if not provider_classes:
        return
    ordered_classes = list(provider_classes)
    warmup_results = await asyncio.gather(
        *(cls.warmup(settings=settings) for cls in ordered_classes),
        return_exceptions=True,
    )
    for cls, res in zip(ordered_classes, warmup_results, strict=False):
        if isinstance(res, BaseException):
            logger.warning("provider_warmup_failed", provider=cls.__name__, exc_info=res)


async def _run_phase(
    kind: Literal["stt", "tts"],
    tasks: Sequence[tuple[RegisteredModel, Awaitable[list[Any]]]],
    completed_item_results: list[Any],
) -> list[Any]:
    """Gather one modality's tasks; a task that raised is logged, not fatal.

    Completed items land in ``completed_item_results`` even if SIGTERM cancels the
    gather. The return value is phase-scoped so an interrupted sample is never read
    as a provider's whole workload by the dead-provider alert.
    """

    async def collect(task: Awaitable[list[Any]]) -> list[Any]:
        results = await task
        completed_item_results.extend(results)
        return results

    batch = await asyncio.gather(*(collect(task) for _entry, task in tasks), return_exceptions=True)
    results: list[Any] = []
    for (entry, _task), batch_result in zip(tasks, batch, strict=True):
        if isinstance(batch_result, BaseException):
            logger.warning(
                f"{kind}_task_raised",
                provider=entry.provider,
                model=entry.model,
                exc_info=batch_result,
            )
        else:
            results.extend(batch_result)
    return results


def _tally(
    ctx: _RunContext, all_results: list[Any], completed_item_results: list[Any]
) -> _RunTally:
    typed = [r for r in all_results if isinstance(r, ctx.result_cls)]
    return _RunTally(
        typed_results=typed,
        success_count=sum(1 for r in typed if r.status == ctx.result_status.SUCCESS),
        fail_count=sum(1 for r in typed if r.status == ctx.result_status.FAILED),
        reliability=_provider_reliability(completed_item_results, ctx.result_status),
    )


def _intended_status(ctx: _RunContext, tally: _RunTally) -> Any:
    if tally.total_results == 0 or tally.fail_count == tally.total_results:
        return ctx.run_status.FAILED
    if tally.success_count == tally.total_results:
        return ctx.run_status.SUCCEEDED
    return ctx.run_status.PARTIAL


def _emit_run_finished(
    ctx: _RunContext,
    *,
    status: Any,
    tally: _RunTally,
    finished_at: datetime,
    sigterm: bool = False,
) -> RunSummary:
    duration_s = (finished_at - ctx.started_at).total_seconds()
    log = logger.warning if sigterm else logger.info
    log(
        "benchmark_run_finished_early_sigterm" if sigterm else "benchmark_run_finished",
        status=str(status),
        total_results=tally.total_results,
        success_count=tally.success_count,
        fail_count=tally.fail_count,
        duration_s=duration_s,
        reliability=[entry.model_dump() for entry in tally.reliability],
    )
    telemetry.record_run(kind=ctx.benchmark_kind, status=str(status))
    properties: dict[str, Any] = {
        "status": str(status),
        "total_results": tally.total_results,
        "success_count": tally.success_count,
        "fail_count": tally.fail_count,
        "duration_s": duration_s,
        "benchmark_kind": ctx.benchmark_kind,
        "smoke": ctx.smoke,
        "$process_person_profile": False,
    }
    if sigterm:
        properties["sigterm"] = True
    _emit_posthog(ctx.posthog_client, "benchmark_run_completed", properties)
    return RunSummary(
        run_id=ctx.run_id,
        started_at=ctx.started_at,
        finished_at=finished_at,
        status=str(status),
        total_results=tally.total_results,
        success_count=tally.success_count,
        fail_count=tally.fail_count,
        reliability=tally.reliability,
        sigterm=sigterm,
    )


async def _seal_run(
    ctx: _RunContext,
    tally: _RunTally,
    *,
    capture_outcomes: list[Any],
    expected_capture_ids: list[str],
) -> RunSummary:
    """Store the final row, then report and refresh.

    Reporting comes after the store on purpose: if finish_run raises, the outer
    handler is the only thing that should page, or one incident pages twice.
    """
    from coval_bench.runner.capture import RunSeal
    from coval_bench.runner.persistence import capture_pending, seal_and_finish_run

    intended_status = _intended_status(ctx, tally)
    incomplete = capture_pending(capture_outcomes, expected_capture_ids)
    held = incomplete and intended_status is not ctx.run_status.FAILED
    finished_at = datetime.now(tz=UTC)
    final_status = await seal_and_finish_run(
        writer=ctx.writer,
        storage_client=ctx.artifact_client,
        bucket=ctx.settings.benchmark_artifact_bucket,
        seal=RunSeal(
            run_id=ctx.run_id,
            intended_status=str(intended_status),
            stored_status=str(ctx.run_status.PARTIAL if held else intended_status),
            finished_at=finished_at,
            error=ctx.capture_pending_error if held else None,
            expected_capture_ids=sorted(expected_capture_ids),
        ),
        capture_pending=incomplete,
    )
    _log_run_outcome(
        final_status,
        tally.typed_results,
        tally.fail_count,
        result_status=ctx.result_status,
        run_status=ctx.run_status,
    )
    if final_status in (ctx.run_status.SUCCEEDED, ctx.run_status.PARTIAL):
        await _refresh_series_bucket(ctx.writer, ctx.run_id)
        # Failed runs retain their queued repairs for hourly maintenance.
        try:
            snapshot_status = await ctx.writer.refresh_window_views(ctx.run_id)
            logger.info("dashboard_windows_refresh", status=snapshot_status)
        except Exception:
            logger.warning("dashboard_windows_refresh_failed", exc_info=True)
    return _emit_run_finished(ctx, status=final_status, tally=tally, finished_at=finished_at)


async def _finish_after_sigterm(ctx: _RunContext, tally: _RunTally) -> RunSummary:
    """Write PARTIAL so the run shows in /v1/results, then report what landed.

    Report before the rollup refresh: SIGTERM gives ~10s, and losing the page
    beats losing the rollup. Only phases that finished before the signal are
    counted; each task already persisted its own rows.
    """
    partial = ctx.run_status.PARTIAL
    try:
        await asyncio.shield(
            ctx.writer.finish_run(
                ctx.run_id,
                status=partial,
                error="sigterm-received (cloud-run-task-timeout)",
            )
        )
    except Exception as write_exc:
        logger.error("run_row_update_failed_after_sigterm", exc_info=write_exc)
    else:
        _log_run_outcome(
            partial,
            tally.typed_results,
            tally.fail_count,
            result_status=ctx.result_status,
            run_status=ctx.run_status,
        )
        await asyncio.shield(_refresh_series_bucket(ctx.writer, ctx.run_id))
    return _emit_run_finished(
        ctx, status=partial, tally=tally, finished_at=datetime.now(tz=UTC), sigterm=True
    )


async def _finish_after_failure(ctx: _RunContext, exc: Exception) -> None:
    """Mark the row FAILED and emit RUN_FAILED, which the Cloud Logging alert keys on."""
    err_msg = _truncate(str(exc))
    log_run_failed(err_msg, exc)
    telemetry.record_run(kind=ctx.benchmark_kind, status=str(ctx.run_status.FAILED))
    try:
        await ctx.writer.finish_run(ctx.run_id, status=ctx.run_status.FAILED, error=err_msg)
    except Exception as write_exc:
        logger.error("run_row_update_failed_after_failure", exc_info=write_exc)
    _emit_posthog(
        ctx.posthog_client,
        "benchmark_run_failed",
        {
            "benchmark_kind": ctx.benchmark_kind,
            "smoke": ctx.smoke,
            "$process_person_profile": False,
        },
    )


async def _run_phases(
    ctx: _RunContext,
    *,
    source: Literal["shared", "dedicated"],
    scheduled_at: datetime,
    load_dataset: Any,
    plan: _DatasetPlan,
    dataset_sha256: str,
    tts_manifest_sha256: str,
    enabled_stt: list[RegisteredModel],
    enabled_tts: list[RegisteredModel],
    all_results: list[Any],
    completed_item_results: list[Any],
) -> RunSummary:
    """Load, warm up, dispatch STT then TTS, then seal the run."""
    settings = ctx.settings
    run_stt = ctx.benchmark_kind in ("stt", "both") and bool(enabled_stt)
    run_tts = ctx.benchmark_kind in ("tts", "both") and bool(enabled_tts)
    stt_items: list[Any] = []
    tts_items: list[Any] = []
    tts_voices: dict[tuple[str, str], list[str | None]] = {}
    expected_capture_ids: list[str] = []
    capture_outcomes: list[Any] = []

    if run_stt:
        stt_dataset = load_dataset(
            plan.stt_dataset_id,
            settings=settings,
            sample_size=None if ctx.smoke else plan.stt_size,
            rng=_get_family_rng()(plan.stt_dataset_id, scheduled_at),
        )
        stt_items = stt_dataset.items[:1] if ctx.smoke else stt_dataset.items
        expected_capture_ids.extend(
            _stt_capture_ids(ctx.run_id, plan.stt_dataset_id, stt_items, enabled_stt)
        )
    if run_tts:
        tts_dataset = load_dataset(
            plan.tts_dataset_id,
            settings=settings,
            sample_size=None if ctx.smoke else plan.tts_size,
        )
        tts_items = tts_dataset.items[:1] if ctx.smoke else tts_dataset.items
        tts_voices = {
            (entry.provider, entry.model): _assign_tts_voices(entry, len(tts_items), ctx.run_id)
            for entry in enabled_tts
        }
        expected_capture_ids.extend(
            _tts_capture_ids(ctx.run_id, plan.tts_dataset_id, tts_items, enabled_tts, tts_voices)
        )
    datasets: dict[str, str] = {}
    if ctx.benchmark_kind in ("stt", "both"):
        datasets[plan.stt_dataset_id] = dataset_sha256
    if ctx.benchmark_kind in ("tts", "both"):
        datasets[plan.tts_dataset_id] = tts_manifest_sha256
    await _upload_manifest(
        ctx,
        scheduled_at=scheduled_at,
        source=source,
        datasets=datasets,
        expected_capture_ids=expected_capture_ids,
    )
    await _warm_providers(settings, ctx.benchmark_kind, enabled_stt, enabled_tts)

    gate = ModelGate(_CONCURRENCY_CAP)
    # Item-major order so the gate interleaves providers; a model-major order
    # would queue one model's items back to back while the others sit idle.
    if run_stt:
        logger.info("stt_dataset_sampled", item_count=len(stt_items))
        stt_tasks = [
            (
                entry,
                _run_stt_item(
                    entry=entry,
                    item=item,
                    run_id=ctx.run_id,
                    gate=gate,
                    settings=settings,
                    writer=ctx.writer,
                    dataset_id=plan.stt_dataset_id,
                    dataset_sha256=dataset_sha256,
                    artifact_client=ctx.artifact_client,
                    capture_outcomes=capture_outcomes,
                ),
            )
            for item in stt_items
            for entry in enabled_stt
        ]
        all_results.extend(await _run_phase("stt", stt_tasks, completed_item_results))
    if run_tts:
        logger.info("tts_dataset_sampled", item_count=len(tts_items))
        tts_tasks = [
            (
                entry,
                _run_tts_item(
                    entry=entry,
                    item=item,
                    run_id=ctx.run_id,
                    gate=gate,
                    settings=settings,
                    voice=tts_voices[(entry.provider, entry.model)][i],
                    writer=ctx.writer,
                    dataset_id=plan.tts_dataset_id,
                    dataset_sha256=tts_manifest_sha256,
                    artifact_client=ctx.artifact_client,
                    capture_outcomes=capture_outcomes,
                ),
            )
            for i, item in enumerate(tts_items)
            for entry in enabled_tts
        ]
        all_results.extend(await _run_phase("tts", tts_tasks, completed_item_results))

    return await _seal_run(
        ctx,
        _tally(ctx, all_results, completed_item_results),
        capture_outcomes=capture_outcomes,
        expected_capture_ids=expected_capture_ids,
    )


async def run_benchmarks(
    *,
    settings: Settings,
    benchmark_kind: Literal["stt", "tts", "both"] = "both",
    smoke: bool = False,
    source: Literal["shared", "dedicated"] = "shared",
    matrix_overrides: list[RegisteredModel] | None = None,
    dataset_id: str | None = None,
    scheduled_at: datetime | None = None,
    only: frozenset[str] | None = None,
) -> RunSummary:
    """Execute one complete benchmark run.

    Args:
        settings: Injected application settings (no global state).
        benchmark_kind: Which benchmark(s) to run.
        smoke: If True, process only the first dataset item (local dev mode).
        source: ``shared`` runs every non-dedicated endpoint; ``dedicated``
            runs only dedicated-inference endpoints. The two never mix in one
            run — dedicated endpoints have their own scheduled job.
        matrix_overrides: Optional list of ``RegisteredModel`` objects that
            override the registry by ``(benchmark, provider, model)`` key.
        dataset_id: STT dataset for this run; defaults to ``settings.dataset_id``.
        scheduled_at: Tick the run is attributed to; suite runs share one.
        only: ``provider/model`` selectors; when set, only those collected models run.

    Returns:
        A :class:`RunSummary` with final counts and run status.

    Raises:
        Exception: On unrecoverable errors (dataset integrity failure, DB issues).
            The run row is updated to ``FAILED`` before re-raising so the Cloud
            Run Job exits non-zero and the log-based metric fires.
    """
    structlog.contextvars.clear_contextvars()

    from coval_bench.config import require_capture_bucket

    require_capture_bucket(settings)

    lifespan_pool, RunWriter, RunStatus, models_mod = _get_db_symbols()
    load_dataset = _get_load_dataset()
    plan = _plan_datasets(settings, source=source, dataset_id=dataset_id, only=only)
    posthog_client = _init_posthog(settings)

    async with lifespan_pool(settings) as pool:
        writer = RunWriter(pool)
        stt_matrix, tts_matrix = _apply_matrix_overrides(await fetch_models(pool), matrix_overrides)
        enabled_stt, enabled_tts = _select_models(
            stt_matrix, tts_matrix, source=source, benchmark_kind=benchmark_kind, only=only
        )

        # A TTS-only run never touches the configured STT dataset; a 'both'
        # run's row still records the STT id. Storage carries the
        # TTS dataset id per row.
        run_dataset_id = plan.tts_dataset_id if benchmark_kind == "tts" else plan.stt_dataset_id
        dataset_sha256 = _manifest_sha256(run_dataset_id, "unknown")
        tts_manifest_sha256 = (
            _manifest_sha256(plan.tts_dataset_id, "") if benchmark_kind in ("tts", "both") else ""
        )
        if scheduled_at is None:
            scheduled_at = _current_tick(settings)
        artifact_client = _capture_client()
        _require_dataset_hashes(benchmark_kind, dataset_sha256, tts_manifest_sha256)
        from coval_bench.runner.capture import preflight_capture_storage

        await asyncio.to_thread(
            preflight_capture_storage,
            artifact_client,
            settings.benchmark_artifact_bucket,
        )
        await writer.preflight_required_capture_schema()
        run = await writer.start_run(
            dataset_id=run_dataset_id,
            dataset_sha256=dataset_sha256,
            scheduled_at=scheduled_at,
        )
        run_id = run.id
        assert run_id is not None  # noqa: S101 — DB always returns an id after INSERT
        ctx = _RunContext(
            settings=settings,
            writer=writer,
            run_id=run_id,
            started_at=run.started_at or datetime.now(tz=UTC),
            artifact_client=artifact_client,
            posthog_client=posthog_client,
            benchmark_kind=benchmark_kind,
            smoke=smoke,
            result_cls=models_mod.Result,
            result_status=models_mod.ResultStatus,
            run_status=RunStatus,
            capture_pending_error=models_mod.CAPTURE_PENDING_ERROR,
        )

        structlog.contextvars.bind_contextvars(run_id=run_id)
        logger.info(
            "benchmark_run_started", benchmark_kind=benchmark_kind, smoke=smoke, source=source
        )

        all_results: list[Any] = []
        completed_item_results: list[Any] = []
        sigterm = _SigtermGuard()
        sigterm.install()
        try:
            return await _run_phases(
                ctx,
                source=source,
                scheduled_at=scheduled_at,
                load_dataset=load_dataset,
                plan=plan,
                dataset_sha256=dataset_sha256,
                tts_manifest_sha256=tts_manifest_sha256,
                enabled_stt=enabled_stt,
                enabled_tts=enabled_tts,
                all_results=all_results,
                completed_item_results=completed_item_results,
            )
        except asyncio.CancelledError:
            # Any CancelledError other than SIGTERM's (e.g. parent shutdown)
            # propagates normally.
            if not sigterm.received:
                raise
            sigterm.uncancel()
            return await _finish_after_sigterm(
                ctx, _tally(ctx, all_results, completed_item_results)
            )
        except Exception as exc:
            await _finish_after_failure(ctx, exc)
            raise
        finally:
            sigterm.uninstall()
            with contextlib.suppress(Exception):
                await _close_http_clients()
            _shutdown_posthog(posthog_client)
