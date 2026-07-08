# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coval_bench.runner.orchestrator.

All external dependencies (providers, dataset, DB) are replaced with mocks /
stubs. NO real network calls are made.

Test catalogue
--------------
1.  test_smoke_run_stt             — 1-item dataset, happy path, SUCCEEDED
2.  test_partial_run               — half providers fail → PARTIAL
3.  test_full_failure              — all providers fail → FAILED
4.  test_retry_succeeds_on_third   — provider fails twice, succeeds 3rd → SUCCESS
5.  test_retry_exhausted           — provider always fails → fail count +1
6.  test_concurrency_cap           — 50 items × 1 provider, ≤8 in flight
7.  test_dataset_integrity_failure — DatasetIntegrityError → run FAILED, re-raised
8.  test_audio_file_cleanup        — TTS audio deleted even when WER raises
9.  test_matrix_overrides          — nova-3 disabled via override → not called
10. test_disabled_providers_skipped — non-ACTIVE entries not instantiated
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import wave
from collections.abc import AsyncIterator, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
import structlog
from posthog import Posthog
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.db.models import Benchmark, Result, ResultStatus, Run, RunStatus
from coval_bench.providers.base import TranscriptionResult, TTSResult
from coval_bench.registries import MODEL_REGISTRY, ModelStatus, RegisteredModel
from coval_bench.runner.orchestrator import RunSummary, run_benchmarks
from coval_bench.runner.retry import with_retry

# ---------------------------------------------------------------------------
# Shared test settings
# ---------------------------------------------------------------------------

_TEST_SETTINGS = Settings(
    database_url="postgresql://runner:password@localhost:5432/benchmarks",
    dataset_bucket="test-bucket",
    dataset_id="stt-v1",
    runner_sha="test-sha",
    log_level="DEBUG",
    openai_api_key=SecretStr("sk-test"),
    deepgram_api_key=SecretStr("dg-test"),
    posthog_disabled=True,
)


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------


def _stt_entry(provider: str, model: str, *, active: bool = True) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.STT,
        provider=provider,
        model=model,
        status=ModelStatus.ACTIVE if active else ModelStatus.PAUSED,
    )


def _tts_entry(provider: str, model: str, voice: str, *, active: bool = True) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.TTS,
        provider=provider,
        model=model,
        voice=voice,
        status=ModelStatus.ACTIVE if active else ModelStatus.PAUSED,
    )


def _registry_entry(benchmark: Benchmark, provider: str) -> RegisteredModel:
    """First registered model for *provider* in *benchmark*; explicit error if missing."""
    for m in MODEL_REGISTRY:
        if m.benchmark is benchmark and m.provider == provider:
            return m
    raise AssertionError(f"no registered {benchmark} model for provider {provider!r}")


def _paused_registry(benchmark: Benchmark) -> list[RegisteredModel]:
    """Pause every registered model for *benchmark*, as an override base."""
    return [
        m.model_copy(update={"status": ModelStatus.PAUSED})
        for m in MODEL_REGISTRY
        if m.benchmark is benchmark
    ]


def _make_run(run_id: int = 1) -> Run:
    return Run(
        id=run_id,
        started_at=datetime.now(tz=UTC),
        finished_at=None,
        runner_sha="test-sha",
        dataset_id="stt-v1",
        dataset_sha256="abc123",
        status=RunStatus.RUNNING,
        error=None,
    )


def _make_dataset_item(path: Path, transcript: str = "hello world") -> Any:
    """Return a namespace acting as a DatasetItem."""
    item = MagicMock()
    item.path = path
    item.transcript = transcript
    item.duration_sec = 1.0
    item.sha256 = "abc"
    item.speech_end_offset_ms = 100.0
    item.metadata = {}
    return item


def _make_tts_item(transcript: str = "hello world") -> Any:
    item = MagicMock()
    item.testcase_id = "tts-0001"
    item.transcript = transcript
    return item


def _good_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        provider="deepgram",
        ttft_seconds=0.25,
        total_time=0.9,
        audio_to_final_seconds=0.85,
        first_token_content="hello",  # noqa: S106 — not a password
        complete_transcript="hello world",
    )


def _make_stub_writer(run: Run) -> MagicMock:
    writer = MagicMock()
    writer.start_run = AsyncMock(return_value=run)
    writer.record_results = AsyncMock()
    writer.finish_run = AsyncMock()
    writer.refresh_stats_matviews = AsyncMock()
    writer.refresh_bucket = AsyncMock()
    return writer


# ---------------------------------------------------------------------------
# Context manager that stubs all sibling modules via orchestrator's helpers
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _orchestrator_env(  # noqa: ANN202
    *,
    audio_path: Path,
    stt_items: list[Any] | None = None,
    tts_items: list[Any] | None = None,
    stt_providers: dict[str, Any] | None = None,
    tts_providers: dict[str, Any] | None = None,
    run: Run | None = None,
    writer: MagicMock | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stub all sibling modules accessed through orchestrator's lazy-import helpers.

    The orchestrator uses ``_get_db_symbols()``, ``_get_stt_providers()``,
    ``_get_tts_providers()``, and ``_get_load_dataset()`` to resolve dependencies
    at call time.  We patch these helpers directly.
    """
    if run is None:
        run = _make_run()
    if writer is None:
        writer = _make_stub_writer(run)
    if stt_items is None:
        stt_items = [_make_dataset_item(audio_path)]
    if tts_items is None:
        tts_items = [_make_tts_item()]
    if stt_providers is None:
        stt_providers = {}
    if tts_providers is None:
        tts_providers = {}

    for cls in (*stt_providers.values(), *tts_providers.values()):
        if not hasattr(cls, "warmup") or not isinstance(cls.warmup, AsyncMock):
            cls.warmup = AsyncMock(return_value=None)

    stt_dataset = MagicMock()
    stt_dataset.items = stt_items
    tts_dataset = MagicMock()
    tts_dataset.items = tts_items

    def _load_dataset(dataset_id: str, *, settings: Any, sample_size: int | None = None) -> Any:
        return stt_dataset if dataset_id == "stt-v1" else tts_dataset

    fake_pool = MagicMock()

    @contextlib.asynccontextmanager
    async def _fake_lifespan_pool(s: Any) -> AsyncIterator[MagicMock]:
        yield fake_pool

    # models_mod stub
    models_mod = MagicMock()
    models_mod.Benchmark = Benchmark
    models_mod.Result = Result
    models_mod.ResultStatus = ResultStatus
    models_mod.RunStatus = RunStatus

    def _fake_get_db_symbols() -> tuple[Any, Any, Any, Any]:
        return _fake_lifespan_pool, MagicMock(return_value=writer), RunStatus, models_mod

    compute_wer_real = __import__("coval_bench.metrics", fromlist=["compute_wer"]).compute_wer
    compute_rtf_real = __import__("coval_bench.metrics", fromlist=["compute_rtf"]).compute_rtf

    with (
        patch(
            "coval_bench.runner.orchestrator._get_db_symbols",
            side_effect=_fake_get_db_symbols,
        ),
        patch(
            "coval_bench.runner.orchestrator._get_stt_providers",
            return_value=stt_providers,
        ),
        patch(
            "coval_bench.runner.orchestrator._get_tts_providers",
            return_value=tts_providers,
        ),
        patch(
            "coval_bench.runner.orchestrator._get_load_dataset",
            return_value=_load_dataset,
        ),
        patch(
            "coval_bench.runner.orchestrator._get_metrics",
            return_value=(compute_wer_real, compute_rtf_real),
        ),
    ):
        yield {"writer": writer, "stt_dataset": stt_dataset, "tts_dataset": tts_dataset}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """Tiny valid 16 kHz mono PCM16 WAV."""
    f = tmp_path / "0001.wav"
    with wave.open(str(f), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 1024)
    return f


@pytest.fixture
def settings() -> Settings:
    return _TEST_SETTINGS


# ---------------------------------------------------------------------------
# 1. test_smoke_run_stt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_run_stt(audio_file: Path, settings: Settings) -> None:
    """1-item dataset, 2 STT providers, happy path → SUCCEEDED."""
    good = _good_transcription()

    provider_a = MagicMock()
    provider_a.measure_ttft = AsyncMock(return_value=good)
    provider_cls_a = MagicMock(return_value=provider_a)

    provider_b = MagicMock()
    provider_b.measure_ttft = AsyncMock(return_value=good)
    provider_cls_b = MagicMock(return_value=provider_b)

    stt_providers = {"deepgram": provider_cls_a, "elevenlabs": provider_cls_b}
    items = [_make_dataset_item(audio_file)]
    matrix = [
        _stt_entry("deepgram", "nova-2"),
        _stt_entry("elevenlabs", "scribe_v2_realtime"),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=items,
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        summary: RunSummary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    assert summary.status == str(RunStatus.SUCCEEDED)
    assert summary.run_id == 1
    # 2 providers × 1 item → at least 3 Result rows per provider (TTFT, AudioToFinal, RTF)
    assert summary.total_results >= 2 * 3
    assert summary.success_count == summary.total_results
    assert summary.fail_count == 0
    writer.start_run.assert_awaited_once()
    # Per-task incremental flush — one call per (registered) provider × item.
    # The default STT matrix has additional deepgram models (nova-3, flux), so
    # both deepgram and elevenlabs each fire multiple times via the same mock.
    assert writer.record_results.await_count >= 2
    writer.finish_run.assert_awaited_once_with(1, status=RunStatus.SUCCEEDED, error=None)
    writer.refresh_stats_matviews.assert_awaited_once_with()
    writer.refresh_bucket.assert_awaited_once_with(
        1, period_seconds=settings.schedule_period_seconds
    )


# ---------------------------------------------------------------------------
# 2. test_partial_run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_run(audio_file: Path, settings: Settings) -> None:
    """Half providers fail → PARTIAL status."""
    good = _good_transcription()

    provider_ok = MagicMock()
    provider_ok.measure_ttft = AsyncMock(return_value=good)
    provider_cls_ok = MagicMock(return_value=provider_ok)

    provider_bad = MagicMock()
    provider_bad.measure_ttft = AsyncMock(side_effect=RuntimeError("boom"))
    provider_cls_bad = MagicMock(return_value=provider_bad)

    stt_providers = {"deepgram": provider_cls_ok, "elevenlabs": provider_cls_bad}
    matrix = [
        _stt_entry("deepgram", "nova-2"),
        _stt_entry("elevenlabs", "scribe_v2_realtime"),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    assert summary.status == str(RunStatus.PARTIAL)
    assert summary.fail_count >= 1
    assert summary.success_count >= 1
    writer.finish_run.assert_awaited_once_with(1, status=RunStatus.PARTIAL, error=None)
    writer.refresh_bucket.assert_awaited_once_with(
        1, period_seconds=settings.schedule_period_seconds
    )


# ---------------------------------------------------------------------------
# 3. test_full_failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_failure(audio_file: Path, settings: Settings) -> None:
    """All providers fail → FAILED status."""
    provider_bad = MagicMock()
    provider_bad.measure_ttft = AsyncMock(side_effect=RuntimeError("always fails"))
    provider_cls_bad = MagicMock(return_value=provider_bad)

    stt_providers = {"deepgram": provider_cls_bad}
    # Override the full registry so only nova-2 deepgram runs
    matrix = [
        *_paused_registry(Benchmark.STT),
        _stt_entry("deepgram", "nova-2"),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    assert summary.status == str(RunStatus.FAILED)
    # Unified failure model: a raised exception fails every metric row for the item
    # (TTFT, AudioToFinal, RTF, TTFS), each carrying the real exception string.
    assert summary.fail_count == 4
    assert summary.success_count == 0
    rows = _recorded_rows(writer)
    assert {r.metric_type for r in rows} == {"TTFT", "AudioToFinal", "RTF", "TTFS"}
    assert all(r.status == ResultStatus.FAILED for r in rows)
    assert all("always fails" in (r.error or "") for r in rows)
    writer.refresh_bucket.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_series_bucket_retries_transient_failure(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient refresh failure is retried; success stops the loop."""
    from coval_bench.runner import orchestrator

    monkeypatch.setattr(orchestrator, "_BUCKET_REFRESH_RETRY_DELAY_S", 0.0)
    writer = MagicMock()
    writer.refresh_bucket = AsyncMock(side_effect=[RuntimeError("blip"), None])

    await orchestrator._refresh_series_bucket(writer, 1, settings)

    assert writer.refresh_bucket.await_count == 2


@pytest.mark.asyncio
async def test_refresh_series_bucket_never_raises(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausting all attempts logs and returns instead of raising."""
    from coval_bench.runner import orchestrator

    monkeypatch.setattr(orchestrator, "_BUCKET_REFRESH_RETRY_DELAY_S", 0.0)
    writer = MagicMock()
    writer.refresh_bucket = AsyncMock(side_effect=RuntimeError("db down"))

    await orchestrator._refresh_series_bucket(writer, 1, settings)

    assert writer.refresh_bucket.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "model"),
    [
        ("deepgram", "flux-general-en"),
        ("deepgram", "flux-general-multi"),
        ("assemblyai", "universal-streaming"),
        ("assemblyai", "universal-streaming-multilingual"),
    ],
)
async def test_non_finalizing_models_excluded_from_ttfs(
    provider_name: str, model: str, audio_file: Path, settings: Settings
) -> None:
    """Models that don't finalize on our end-of-speech signal get no TTFS row.

    Flux has no client finalize; the AssemblyAI universal-streaming models ack
    ForceEndpoint without flushing the tail. The other metrics still run.
    """
    provider = MagicMock()
    provider.measure_ttft = AsyncMock(return_value=_good_transcription())
    stt_providers = {provider_name: MagicMock(return_value=provider)}
    matrix = [
        *_paused_registry(Benchmark.STT),
        _stt_entry(provider_name, model),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    metric_types = {r.metric_type for r in _recorded_rows(writer)}
    assert "TTFS" not in metric_types
    assert metric_types == {"TTFT", "AudioToFinal", "RTF", "WER"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "model"),
    [
        ("xai", "grok-stt"),
        ("openai", "gpt-4o-transcribe"),
        ("openai", "gpt-4o-mini-transcribe"),
    ],
)
async def test_non_streaming_models_excluded_from_ttft(
    provider_name: str, model: str, audio_file: Path, settings: Settings
) -> None:
    """Models that can't emit a token before end-of-speech get no TTFT row.

    xAI Grok (min-audio gate) and the OpenAI transcribe models (finalize-only, no
    mid-utterance partials) report TTFS instead; the other metrics still run.
    """
    provider = MagicMock()
    provider.measure_ttft = AsyncMock(return_value=_good_transcription())
    stt_providers = {provider_name: MagicMock(return_value=provider)}
    matrix = [
        *_paused_registry(Benchmark.STT),
        _stt_entry(provider_name, model),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    metric_types = {r.metric_type for r in _recorded_rows(writer)}
    assert "TTFT" not in metric_types
    assert metric_types == {"AudioToFinal", "RTF", "TTFS", "WER"}


# ---------------------------------------------------------------------------
# 4. test_retry_succeeds_on_third_attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_on_third_attempt() -> None:
    """with_retry calls fn 3 times (fails twice, succeeds 3rd)."""
    call_count = 0

    async def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError  # asyncio.TimeoutError is an alias for builtin TimeoutError
        return "ok"

    result = await with_retry(
        flaky,
        max_attempts=3,
        base_delay_s=0.0,
        max_delay_s=0.0,
    )
    assert result == "ok"
    assert call_count == 3


# ---------------------------------------------------------------------------
# 5. test_retry_exhausted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_exhausted() -> None:
    """with_retry exhausts 3 attempts then re-raises."""
    call_count = 0

    async def always_fails() -> str:
        nonlocal call_count
        call_count += 1
        raise TimeoutError("timeout")

    with pytest.raises(TimeoutError):
        await with_retry(
            always_fails,
            max_attempts=3,
            base_delay_s=0.0,
            max_delay_s=0.0,
        )

    assert call_count == 3


# ---------------------------------------------------------------------------
# 6. test_concurrency_cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_cap(audio_file: Path, settings: Settings) -> None:
    """50 dataset items × 1 provider — at most 8 coroutines in flight at once."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def tracked_measure_ttft(*args: Any, **kwargs: Any) -> TranscriptionResult:
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
        await asyncio.sleep(0)  # yield to event loop
        async with lock:
            current_concurrent -= 1
        return _good_transcription()

    provider_inst = MagicMock()
    provider_inst.measure_ttft = tracked_measure_ttft
    provider_cls = MagicMock(return_value=provider_inst)

    stt_providers = {"deepgram": provider_cls}
    items_50 = [_make_dataset_item(audio_file, f"item {i}") for i in range(50)]
    matrix = [_stt_entry("deepgram", "nova-2")]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=items_50,
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=False,
            matrix_overrides=matrix,
        )

    assert max_concurrent <= 8, f"max concurrent was {max_concurrent}"
    assert summary.total_results >= 50 * 3


# ---------------------------------------------------------------------------
# 7. test_dataset_integrity_failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dataset_integrity_failure(settings: Settings) -> None:
    """DatasetIntegrityError → run marked FAILED, exception re-raised."""

    class DatasetIntegrityError(Exception):
        pass

    def _bad_load(dataset_id: str, *, settings: Any, sample_size: int | None = None) -> Any:
        raise DatasetIntegrityError("hash mismatch: expected abc actual def")

    run = _make_run()
    writer = _make_stub_writer(run)

    fake_pool = MagicMock()

    @contextlib.asynccontextmanager
    async def _fake_lifespan_pool(s: Any) -> AsyncIterator[MagicMock]:
        yield fake_pool

    models_mod = MagicMock()
    models_mod.Benchmark = Benchmark
    models_mod.Result = Result
    models_mod.ResultStatus = ResultStatus
    models_mod.RunStatus = RunStatus

    def _fake_get_db_symbols() -> tuple[Any, Any, Any, Any]:
        return _fake_lifespan_pool, MagicMock(return_value=writer), RunStatus, models_mod

    deepgram_cls = MagicMock()
    deepgram_cls.warmup = AsyncMock(return_value=None)

    with (
        patch(
            "coval_bench.runner.orchestrator._get_db_symbols",
            side_effect=_fake_get_db_symbols,
        ),
        patch(
            "coval_bench.runner.orchestrator._get_stt_providers",
            return_value={"deepgram": deepgram_cls},
        ),
        patch(
            "coval_bench.runner.orchestrator._get_tts_providers",
            return_value={},
        ),
        patch(
            "coval_bench.runner.orchestrator._get_load_dataset",
            return_value=_bad_load,
        ),
        patch(
            "coval_bench.runner.orchestrator._get_metrics",
            return_value=(MagicMock(), MagicMock()),
        ),
        pytest.raises(DatasetIntegrityError),
    ):
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            matrix_overrides=[_stt_entry("deepgram", "nova-2")],
        )

    finish_call = writer.finish_run.call_args
    assert finish_call.kwargs["status"] == RunStatus.FAILED
    assert finish_call.kwargs["error"] is not None


# ---------------------------------------------------------------------------
# 8. test_audio_file_cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_file_cleanup(settings: Settings) -> None:
    """TTS audio file is deleted even when WER (_transcribe_with_whisper) raises."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "synth.wav"
        audio_path.write_bytes(b"\x00" * 512)

        tts_result = TTSResult(
            provider="elevenlabs",
            model="eleven_flash_v2_5",
            voice="IKne3meq5aSn9XLyUdCD",
            ttfa_ms=120.0,
            audio_path=audio_path,
            error=None,
        )

        provider_inst = MagicMock()
        provider_inst.synthesize = AsyncMock(return_value=tts_result)
        provider_cls = MagicMock(return_value=provider_inst)
        provider_cls.warmup = AsyncMock(return_value=None)

        tts_providers = {"elevenlabs": provider_cls}
        tts_item = _make_tts_item("hello world")

        run = _make_run()
        writer = _make_stub_writer(run)

        tts_dataset = MagicMock()
        tts_dataset.items = [tts_item]

        def _load(dataset_id: str, *, settings: Any, sample_size: int | None = None) -> Any:
            return tts_dataset

        fake_pool = MagicMock()

        @contextlib.asynccontextmanager
        async def _fake_pool(s: Any) -> AsyncIterator[MagicMock]:
            yield fake_pool

        models_mod = MagicMock()
        models_mod.Benchmark = Benchmark
        models_mod.Result = Result
        models_mod.ResultStatus = ResultStatus
        models_mod.RunStatus = RunStatus

        def _fake_get_db_symbols() -> tuple[Any, Any, Any, Any]:
            return _fake_pool, MagicMock(return_value=writer), RunStatus, models_mod

        compute_wer_real = __import__("coval_bench.metrics", fromlist=["compute_wer"]).compute_wer
        compute_rtf_real = __import__("coval_bench.metrics", fromlist=["compute_rtf"]).compute_rtf

        matrix = [_tts_entry("elevenlabs", "eleven_flash_v2_5", "IKne3meq5aSn9XLyUdCD")]

        with (
            patch(
                "coval_bench.runner.orchestrator._get_db_symbols",
                side_effect=_fake_get_db_symbols,
            ),
            patch(
                "coval_bench.runner.orchestrator._get_stt_providers",
                return_value={},
            ),
            patch(
                "coval_bench.runner.orchestrator._get_tts_providers",
                return_value=tts_providers,
            ),
            patch(
                "coval_bench.runner.orchestrator._get_load_dataset",
                return_value=_load,
            ),
            patch(
                "coval_bench.runner.orchestrator._get_metrics",
                return_value=(compute_wer_real, compute_rtf_real),
            ),
            patch(
                "coval_bench.runner.orchestrator._transcribe_with_whisper",
                side_effect=RuntimeError("whisper unavailable"),
            ),
        ):
            summary = await run_benchmarks(
                settings=settings,
                benchmark_kind="tts",
                smoke=True,
                matrix_overrides=matrix,
            )

        # Audio file must be gone regardless of the WER exception
        assert not audio_path.exists(), "TTS audio file was not cleaned up"
        # TTFA result was still recorded successfully
        assert summary.success_count >= 1


@pytest.mark.asyncio
async def test_tts_http1_downgrade_nulls_ttfa_row(settings: Settings) -> None:
    """HTTP/1.1 TTFA is a null-valued success; WER stays SUCCESS; run stays SUCCEEDED."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "synth.wav"
        audio_path.write_bytes(b"\x00" * 512)

        tts_result = TTSResult(
            provider="elevenlabs",
            model="eleven_flash_v2_5",
            voice="IKne3meq5aSn9XLyUdCD",
            ttfa_ms=120.0,
            audio_path=audio_path,
            error=None,
            http_version="HTTP/1.1",
            submit_to_headers_ms=210.0,
        )

        provider_inst = MagicMock()
        provider_inst.synthesize = AsyncMock(return_value=tts_result)
        provider_cls = MagicMock(return_value=provider_inst)
        provider_cls.warmup = AsyncMock(return_value=None)

        tts_providers = {"elevenlabs": provider_cls}

        run = _make_run()
        writer = _make_stub_writer(run)

        tts_dataset = MagicMock()
        tts_dataset.items = [_make_tts_item("hello world")]

        def _load(dataset_id: str, *, settings: Any, sample_size: int | None = None) -> Any:
            return tts_dataset

        fake_pool = MagicMock()

        @contextlib.asynccontextmanager
        async def _fake_pool(s: Any) -> AsyncIterator[MagicMock]:
            yield fake_pool

        models_mod = MagicMock()
        models_mod.Benchmark = Benchmark
        models_mod.Result = Result
        models_mod.ResultStatus = ResultStatus
        models_mod.RunStatus = RunStatus

        def _fake_get_db_symbols() -> tuple[Any, Any, Any, Any]:
            return _fake_pool, MagicMock(return_value=writer), RunStatus, models_mod

        compute_wer_real = __import__("coval_bench.metrics", fromlist=["compute_wer"]).compute_wer
        compute_rtf_real = __import__("coval_bench.metrics", fromlist=["compute_rtf"]).compute_rtf

        matrix = [
            *_paused_registry(Benchmark.TTS),
            _tts_entry("elevenlabs", "eleven_flash_v2_5", "IKne3meq5aSn9XLyUdCD"),
        ]

        with (
            patch(
                "coval_bench.runner.orchestrator._get_db_symbols",
                side_effect=_fake_get_db_symbols,
            ),
            patch("coval_bench.runner.orchestrator._get_stt_providers", return_value={}),
            patch(
                "coval_bench.runner.orchestrator._get_tts_providers",
                return_value=tts_providers,
            ),
            patch("coval_bench.runner.orchestrator._get_load_dataset", return_value=_load),
            patch(
                "coval_bench.runner.orchestrator._get_metrics",
                return_value=(compute_wer_real, compute_rtf_real),
            ),
            patch(
                "coval_bench.runner.orchestrator._transcribe_with_whisper",
                return_value="hello world",
            ),
        ):
            await run_benchmarks(
                settings=settings,
                benchmark_kind="tts",
                smoke=True,
                matrix_overrides=matrix,
            )

        recorded = [r for call in writer.record_results.call_args_list for r in call.args[0]]
        ttfa_rows = [r for r in recorded if r.metric_type == "TTFA"]
        wer_rows = [r for r in recorded if r.metric_type == "WER"]

        assert len(ttfa_rows) == 1
        assert ttfa_rows[0].status == ResultStatus.SUCCESS
        assert ttfa_rows[0].metric_value is None
        assert "HTTP/1.1" in ttfa_rows[0].error
        assert ttfa_rows[0].http_version == "HTTP/1.1"

        assert len(wer_rows) == 1
        assert wer_rows[0].status == ResultStatus.SUCCESS

        assert writer.finish_run.call_args.kwargs["status"] == RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_tts_cold_connection_nulls_ttfa_row(settings: Settings) -> None:
    """A cold-connection TTFA is a null-valued success, not a PARTIAL-forcing failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "synth.wav"
        audio_path.write_bytes(b"\x00" * 512)

        tts_result = TTSResult(
            provider="elevenlabs",
            model="eleven_flash_v2_5",
            voice="IKne3meq5aSn9XLyUdCD",
            ttfa_ms=120.0,
            audio_path=audio_path,
            error=None,
            http_version="HTTP/2",
            submit_to_headers_ms=190.0,
            connection_reused=False,
        )

        provider_inst = MagicMock()
        provider_inst.synthesize = AsyncMock(return_value=tts_result)
        provider_cls = MagicMock(return_value=provider_inst)
        provider_cls.warmup = AsyncMock(return_value=None)

        tts_providers = {"elevenlabs": provider_cls}

        run = _make_run()
        writer = _make_stub_writer(run)

        tts_dataset = MagicMock()
        tts_dataset.items = [_make_tts_item("hello world")]

        def _load(dataset_id: str, *, settings: Any, sample_size: int | None = None) -> Any:
            return tts_dataset

        fake_pool = MagicMock()

        @contextlib.asynccontextmanager
        async def _fake_pool(s: Any) -> AsyncIterator[MagicMock]:
            yield fake_pool

        models_mod = MagicMock()
        models_mod.Benchmark = Benchmark
        models_mod.Result = Result
        models_mod.ResultStatus = ResultStatus
        models_mod.RunStatus = RunStatus

        def _fake_get_db_symbols() -> tuple[Any, Any, Any, Any]:
            return _fake_pool, MagicMock(return_value=writer), RunStatus, models_mod

        compute_wer_real = __import__("coval_bench.metrics", fromlist=["compute_wer"]).compute_wer
        compute_rtf_real = __import__("coval_bench.metrics", fromlist=["compute_rtf"]).compute_rtf

        matrix = [
            *_paused_registry(Benchmark.TTS),
            _tts_entry("elevenlabs", "eleven_flash_v2_5", "IKne3meq5aSn9XLyUdCD"),
        ]

        with (
            patch(
                "coval_bench.runner.orchestrator._get_db_symbols",
                side_effect=_fake_get_db_symbols,
            ),
            patch("coval_bench.runner.orchestrator._get_stt_providers", return_value={}),
            patch(
                "coval_bench.runner.orchestrator._get_tts_providers",
                return_value=tts_providers,
            ),
            patch("coval_bench.runner.orchestrator._get_load_dataset", return_value=_load),
            patch(
                "coval_bench.runner.orchestrator._get_metrics",
                return_value=(compute_wer_real, compute_rtf_real),
            ),
            patch(
                "coval_bench.runner.orchestrator._transcribe_with_whisper",
                return_value="hello world",
            ),
        ):
            await run_benchmarks(
                settings=settings,
                benchmark_kind="tts",
                smoke=True,
                matrix_overrides=matrix,
            )

        recorded = [r for call in writer.record_results.call_args_list for r in call.args[0]]
        ttfa_rows = [r for r in recorded if r.metric_type == "TTFA"]
        wer_rows = [r for r in recorded if r.metric_type == "WER"]

        assert len(ttfa_rows) == 1
        assert ttfa_rows[0].status == ResultStatus.SUCCESS
        assert ttfa_rows[0].metric_value is None
        assert "cold connection" in ttfa_rows[0].error
        assert ttfa_rows[0].http_version == "HTTP/2"

        assert len(wer_rows) == 1
        assert wer_rows[0].status == ResultStatus.SUCCESS

        assert writer.finish_run.call_args.kwargs["status"] == RunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# 9. test_matrix_overrides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_matrix_overrides(audio_file: Path, settings: Settings) -> None:
    """nova-3 disabled via override → measure_ttft never called for nova-3."""
    good = _good_transcription()

    # Each model gets a distinct instance so we can assert per-model call counts
    model_instances: dict[str, MagicMock] = {}

    def _make_provider(api_key: Any, model: str) -> MagicMock:
        inst = MagicMock()
        inst.measure_ttft = AsyncMock(return_value=good)
        model_instances[model] = inst
        return inst

    stt_providers = {"deepgram": _make_provider}

    # Override: nova-2 active, nova-3 paused; pause all others too
    matrix = [
        *_paused_registry(Benchmark.STT),
        _stt_entry("deepgram", "nova-2"),
        _stt_entry("deepgram", "nova-3", active=False),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    # nova-3 must never have been instantiated or called
    assert "nova-3" not in model_instances, f"nova-3 was instantiated: {list(model_instances)}"
    # nova-2 was instantiated and called
    assert "nova-2" in model_instances
    model_instances["nova-2"].measure_ttft.assert_awaited()


@pytest.mark.asyncio
async def test_warmup_scoped_to_benchmark_kind(audio_file: Path, settings: Settings) -> None:
    """An stt run warms only stt providers; tts warmup is not invoked."""
    good = _good_transcription()

    stt_inst = MagicMock()
    stt_inst.measure_ttft = AsyncMock(return_value=good)
    stt_cls = MagicMock(return_value=stt_inst)
    stt_cls.warmup = AsyncMock(return_value=None)

    tts_cls = MagicMock()
    tts_cls.warmup = AsyncMock(return_value=None)

    matrix = [
        _stt_entry("deepgram", "nova-2"),
        _tts_entry("elevenlabs", "eleven_flash_v2_5", "IKne3meq5aSn9XLyUdCD"),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": stt_cls},
        tts_providers={"elevenlabs": tts_cls},
        run=run,
        writer=writer,
    ) as _:
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    stt_cls.warmup.assert_awaited_once()
    tts_cls.warmup.assert_not_awaited()


# ---------------------------------------------------------------------------
# 10. test_disabled_providers_skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_providers_skipped(audio_file: Path, settings: Settings) -> None:
    """Entries whose status is not ACTIVE are never instantiated or called."""
    good = _good_transcription()

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=good)
    provider_cls = MagicMock(return_value=provider_inst)

    disabled_cls = MagicMock()

    stt_providers = {"deepgram": provider_cls, "elevenlabs": disabled_cls}
    matrix = [
        _stt_entry("deepgram", "nova-2"),
        _stt_entry("elevenlabs", "scribe_v2_realtime", active=False),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    disabled_cls.assert_not_called()
    provider_cls.assert_called()


# ---------------------------------------------------------------------------
# 11b. dataset sampling — one shared sample per run (parity), smoke opts out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_samples_dataset_once_with_configured_size(
    audio_file: Path, settings: Settings
) -> None:
    """A real run loads the dataset once with dataset_sample_size — one shared sample."""
    provider = MagicMock()
    provider.measure_ttft = AsyncMock(return_value=_good_transcription())
    matrix = [
        _stt_entry("deepgram", "nova-2"),
        _stt_entry("elevenlabs", "scribe_v2_realtime"),
    ]
    sized = settings.model_copy(update={"dataset_sample_size": 7})

    items = [_make_dataset_item(audio_file, f"item {i}") for i in range(7)]
    captured: list[int | None] = []

    stt_dataset = MagicMock()
    stt_dataset.items = items

    def _load(dataset_id: str, *, settings: Any, sample_size: int | None = None) -> Any:
        captured.append(sample_size)
        return stt_dataset

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=items,
        stt_providers={
            "deepgram": MagicMock(return_value=provider),
            "elevenlabs": MagicMock(return_value=provider),
        },
        run=run,
        writer=writer,
    ):
        with patch("coval_bench.runner.orchestrator._get_load_dataset", return_value=_load):
            await run_benchmarks(
                settings=sized,
                benchmark_kind="stt",
                smoke=False,
                matrix_overrides=matrix,
            )

    assert captured == [7]


@pytest.mark.asyncio
async def test_smoke_run_skips_sampling(audio_file: Path, settings: Settings) -> None:
    """Smoke mode bypasses sampling (sample_size=None) so local dev is deterministic."""
    provider = MagicMock()
    provider.measure_ttft = AsyncMock(return_value=_good_transcription())
    captured: list[int | None] = []

    stt_dataset = MagicMock()
    stt_dataset.items = [_make_dataset_item(audio_file, "only")]

    def _load(dataset_id: str, *, settings: Any, sample_size: int | None = None) -> Any:
        captured.append(sample_size)
        return stt_dataset

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=stt_dataset.items,
        stt_providers={"deepgram": MagicMock(return_value=provider)},
        run=run,
        writer=writer,
    ):
        with patch("coval_bench.runner.orchestrator._get_load_dataset", return_value=_load):
            await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=[_stt_entry("deepgram", "nova-2")],
            )

    assert captured == [None]


# ---------------------------------------------------------------------------
# 12. test_incremental_flush_persists_completed_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_flush_persists_completed_tasks(
    audio_file: Path, settings: Settings
) -> None:
    """Each provider×item task flushes its own results — completed tasks are
    persisted before later tasks run, so a mid-run cancellation cannot lose
    them. Asserts ordering: ``record_results`` is awaited *during* gather, not
    once at the end after all tasks finish.
    """
    good = _good_transcription()

    # Provider A finishes immediately; provider B sleeps then finishes.
    provider_a = MagicMock()
    provider_a.measure_ttft = AsyncMock(return_value=good)
    provider_cls_a = MagicMock(return_value=provider_a)

    provider_b_started = asyncio.Event()
    provider_b_release = asyncio.Event()

    async def slow_measure(*args: Any, **kwargs: Any) -> TranscriptionResult:
        provider_b_started.set()
        await provider_b_release.wait()
        return good

    provider_b = MagicMock()
    provider_b.measure_ttft = slow_measure
    provider_cls_b = MagicMock(return_value=provider_b)

    stt_providers = {"deepgram": provider_cls_a, "elevenlabs": provider_cls_b}
    matrix = [
        _stt_entry("deepgram", "nova-2"),
        _stt_entry("elevenlabs", "scribe_v2_realtime"),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)
    record_calls_when_b_started: list[int] = []

    original_record = writer.record_results

    async def _record_and_observe(results: Any) -> None:
        # Snapshot how many calls had completed when each call lands.
        record_calls_when_b_started.append(original_record.await_count)
        await original_record(results)

    writer.record_results = AsyncMock(side_effect=_record_and_observe)

    async def _release_after_a_persisted() -> None:
        # Wait until provider A's record_results has been awaited at least once,
        # then release provider B. If persistence were batched at end-of-run,
        # this would deadlock (provider B waits for release, gather waits for B,
        # record_results never fires).
        await provider_b_started.wait()
        while writer.record_results.await_count < 1:
            await asyncio.sleep(0.01)
        provider_b_release.set()

    releaser = asyncio.create_task(_release_after_a_persisted())

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers=stt_providers,
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    await releaser

    assert summary.status == str(RunStatus.SUCCEEDED)
    # At least two tasks → at least two flushes (default matrix has extra
    # deepgram/elevenlabs entries that the override map merges through).
    assert writer.record_results.await_count >= 2
    # The releaser only fired record_results once provider A had been persisted
    # — proving we don't wait for B before flushing A.
    assert provider_b_release.is_set()


# ---------------------------------------------------------------------------
# 13. test_sigterm_finalizes_run_as_partial
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(asyncio.get_event_loop_policy().new_event_loop(), "add_signal_handler"),
    reason="loop.add_signal_handler is POSIX-only",
)
@pytest.mark.asyncio
async def test_sigterm_finalizes_run_as_partial(audio_file: Path, settings: Settings) -> None:
    """SIGTERM mid-run → writer.finish_run called with PARTIAL.

    Cloud Run sends SIGTERM ~10s before SIGKILL when a task hits its timeout.
    Without the handler, the run row stays at status='running' (invisible to
    the API filter that includes only 'succeeded'/'partial'). This regression
    test guards the recovery path: the orchestrator must intercept SIGTERM,
    cancel the in-flight gather, and update the run row to PARTIAL.
    """
    import os
    import signal

    good = _good_transcription()

    started = asyncio.Event()

    async def _slow_measure(*args: Any, **kwargs: Any) -> TranscriptionResult:
        started.set()
        await asyncio.sleep(60)  # would normally hang; SIGTERM short-circuits
        return good

    provider = MagicMock()
    provider.measure_ttft = _slow_measure
    provider_cls = MagicMock(return_value=provider)

    matrix = [_stt_entry("deepgram", "nova-2")]

    run = _make_run()
    writer = _make_stub_writer(run)

    async def _send_sigterm_after_start() -> None:
        await started.wait()
        # Tiny sleep so the gather is suspended at the await before SIGTERM.
        await asyncio.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)

    sigterm_task = asyncio.create_task(_send_sigterm_after_start())

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": provider_cls},
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=matrix,
        )

    await sigterm_task

    assert summary.status == str(RunStatus.PARTIAL)
    writer.finish_run.assert_awaited_once()
    finish_kwargs = writer.finish_run.await_args.kwargs
    assert finish_kwargs["status"] == RunStatus.PARTIAL
    assert "sigterm" in (finish_kwargs.get("error") or "").lower()
    writer.refresh_bucket.assert_awaited_once_with(
        run.id, period_seconds=settings.schedule_period_seconds
    )


# ---------------------------------------------------------------------------
# 14-17. degraded results that return (no raise) are marked FAILED
# ---------------------------------------------------------------------------


def _only_stt_matrix(provider: str, model: str) -> list[RegisteredModel]:
    """Pause the whole STT registry and activate just one provider×model.

    Keeps a degraded-provider test isolated to a single Result set (the
    registry runs deepgram across several models, which would otherwise
    collide).
    """
    return [
        *_paused_registry(Benchmark.STT),
        _stt_entry(provider, model),
    ]


def _recorded_rows(writer: MagicMock) -> list[Result]:
    """All Result rows handed to writer.record_results across the run."""
    return [row for call in writer.record_results.await_args_list for row in call.args[0]]


def _events(captured: list[MutableMapping[str, Any]], name: str) -> list[MutableMapping[str, Any]]:
    """Captured structlog events whose event name matches *name*."""
    return [e for e in captured if e.get("event") == name]


@pytest.mark.asyncio
async def test_stt_empty_result_marked_failed(audio_file: Path, settings: Settings) -> None:
    """No-raise return with no metrics → TTFT/AudioToFinal/RTF FAILED, no WER row, run FAILED."""
    empty = TranscriptionResult(provider="deepgram")  # all metrics None, no error, no transcript

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=empty)
    provider_cls = MagicMock(return_value=provider_inst)

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": provider_cls},
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
        )

    by_metric = {r.metric_type: r for r in _recorded_rows(writer)}
    for mt in ("TTFT", "AudioToFinal", "RTF"):
        assert by_metric[mt].status == ResultStatus.FAILED
        error = by_metric[mt].error
        assert error is not None and "produced" in error
    assert "WER" not in by_metric  # no transcript → no WER row
    assert summary.success_count == 0
    assert summary.status == str(RunStatus.FAILED)


@pytest.mark.asyncio
async def test_stt_result_error_propagated(audio_file: Path, settings: Settings) -> None:
    """result.error set → every row FAILED, provider message preserved, WER skipped."""
    errored = TranscriptionResult(
        provider="deepgram",
        ttft_seconds=0.3,  # a measured TTFT is still untrustworthy when the stream errored
        complete_transcript="partial words",
        error="websocket closed unexpectedly",
    )

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=errored)
    provider_cls = MagicMock(return_value=provider_inst)

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": provider_cls},
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
        )

    rows = _recorded_rows(writer)
    assert rows
    assert all(r.status == ResultStatus.FAILED for r in rows)
    assert all("websocket closed" in (r.error or "") for r in rows)
    assert all(r.metric_type != "WER" for r in rows)  # errored stream → no WER scoring
    assert summary.success_count == 0


@pytest.mark.asyncio
async def test_stt_partial_keeps_real_ttft(audio_file: Path, settings: Settings) -> None:
    """First token but no final → TTFT success, AudioToFinal/RTF FAILED → run PARTIAL."""
    partial = TranscriptionResult(
        provider="deepgram",
        ttft_seconds=0.42,
        audio_to_final_seconds=None,
        complete_transcript=None,
    )

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=partial)
    provider_cls = MagicMock(return_value=provider_inst)

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": provider_cls},
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
        )

    by_metric = {r.metric_type: r for r in _recorded_rows(writer)}
    assert by_metric["TTFT"].status == ResultStatus.SUCCESS
    assert by_metric["TTFT"].metric_value == 0.42
    assert by_metric["AudioToFinal"].status == ResultStatus.FAILED
    assert by_metric["RTF"].status == ResultStatus.FAILED
    assert summary.status == str(RunStatus.PARTIAL)


@pytest.mark.asyncio
async def test_stt_ttfs_status_tracks_value(audio_file: Path, settings: Settings) -> None:
    """TTFS succeeds with audio_to_final − offset when an offset is pinned, and fails
    (never a null-valued success) when the offset is missing."""
    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=_good_transcription())
    run = _make_run()
    writer = _make_stub_writer(run)
    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],  # speech_end_offset_ms = 100.0
        stt_providers={"deepgram": MagicMock(return_value=provider_inst)},
        run=run,
        writer=writer,
    ) as _:
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
        )
    by_metric = {r.metric_type: r for r in _recorded_rows(writer)}
    assert by_metric["TTFS"].status == ResultStatus.SUCCESS
    assert by_metric["TTFS"].metric_value == pytest.approx(0.85 - 0.1)

    item_no_offset = _make_dataset_item(audio_file)
    item_no_offset.speech_end_offset_ms = None
    provider_inst2 = MagicMock()
    provider_inst2.measure_ttft = AsyncMock(return_value=_good_transcription())
    run2 = _make_run()
    writer2 = _make_stub_writer(run2)
    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[item_no_offset],
        stt_providers={"deepgram": MagicMock(return_value=provider_inst2)},
        run=run2,
        writer=writer2,
    ) as _:
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
        )
    by_metric2 = {r.metric_type: r for r in _recorded_rows(writer2)}
    assert by_metric2["TTFS"].metric_value is None
    assert by_metric2["TTFS"].status == ResultStatus.FAILED


@pytest.mark.asyncio
async def test_stt_ttfs_early_final_clamps_to_zero(audio_file: Path, settings: Settings) -> None:
    """A final ahead of the end-of-speech anchor yields a 0.0-valued TTFS success, not a failure."""
    early_item = _make_dataset_item(audio_file)
    early_item.speech_end_offset_ms = 2000.0  # exceeds audio_to_final (0.85 s)
    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=_good_transcription())
    run = _make_run()
    writer = _make_stub_writer(run)
    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[early_item],
        stt_providers={"deepgram": MagicMock(return_value=provider_inst)},
        run=run,
        writer=writer,
    ) as _:
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            smoke=True,
            matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
        )
    ttfs = {r.metric_type: r for r in _recorded_rows(writer)}["TTFS"]
    assert ttfs.status == ResultStatus.SUCCESS
    assert ttfs.metric_value == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_tts_empty_ttfa_marked_failed(audio_file: Path, settings: Settings) -> None:
    """TTS synth returns (no raise) with no ttfa/audio → TTFA FAILED, no WER row, run FAILED."""
    hume_entry = _registry_entry(Benchmark.TTS, "hume")
    empty_tts = TTSResult(
        provider="hume",
        model=hume_entry.model,
        voice=hume_entry.voice or "v",
        ttfa_ms=None,
        audio_path=None,
        error=None,
    )

    provider_inst = MagicMock()
    provider_inst.synthesize = AsyncMock(return_value=empty_tts)
    provider_cls = MagicMock(return_value=provider_inst)

    # Activate exactly one hume entry; pause the rest of the TTS registry.
    matrix = [
        *_paused_registry(Benchmark.TTS),
        hume_entry.model_copy(update={"status": ModelStatus.ACTIVE}),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        tts_items=[_make_tts_item("hello world")],
        tts_providers={"hume": provider_cls},
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="tts",
            smoke=True,
            matrix_overrides=matrix,
        )

    by_metric = {r.metric_type: r for r in _recorded_rows(writer)}
    assert by_metric["TTFA"].status == ResultStatus.FAILED
    assert by_metric["TTFA"].error and "TTFA" in by_metric["TTFA"].error
    assert "WER" not in by_metric
    assert summary.success_count == 0
    assert summary.status == str(RunStatus.FAILED)


@pytest.mark.asyncio
async def test_tts_provider_error_wins_over_contamination(
    audio_file: Path, settings: Settings
) -> None:
    """A provider error on an HTTP/1.1 result fails the row with the provider message.

    Transport contamination only downgrades a would-be SUCCESS; a real provider error keeps
    its own (more specific) message and suppresses WER.
    """
    hume_entry = _registry_entry(Benchmark.TTS, "hume")
    errored_tts = TTSResult(
        provider="hume",
        model=hume_entry.model,
        voice=hume_entry.voice or "v",
        ttfa_ms=120.0,
        audio_path=None,
        error="synth stream closed early",
        http_version="HTTP/1.1",  # would otherwise trigger the contamination message
    )

    provider_inst = MagicMock()
    provider_inst.synthesize = AsyncMock(return_value=errored_tts)
    provider_cls = MagicMock(return_value=provider_inst)

    matrix = [
        *_paused_registry(Benchmark.TTS),
        hume_entry.model_copy(update={"status": ModelStatus.ACTIVE}),
    ]

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        tts_items=[_make_tts_item("hello world")],
        tts_providers={"hume": provider_cls},
        run=run,
        writer=writer,
    ) as _:
        summary = await run_benchmarks(
            settings=settings,
            benchmark_kind="tts",
            smoke=True,
            matrix_overrides=matrix,
        )

    by_metric = {r.metric_type: r for r in _recorded_rows(writer)}
    ttfa = by_metric["TTFA"]
    assert ttfa.status == ResultStatus.FAILED
    assert "synth stream closed early" in (ttfa.error or "")
    assert "HTTP/1.1" not in (ttfa.error or "")  # contamination message must not win
    assert ttfa.http_version == "HTTP/1.1"  # diagnostic still recorded
    assert "WER" not in by_metric  # errored synth → no WER scoring
    assert summary.success_count == 0


@pytest.mark.asyncio
async def test_stt_wer_compute_failure_marked_failed(audio_file: Path, settings: Settings) -> None:
    """compute_wer crashing on a real transcript → FAILED WER row, run PARTIAL (not silent)."""
    from coval_bench.metrics import compute_rtf

    def _raising_wer(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("wer blew up")

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=_good_transcription())
    provider_cls = MagicMock(return_value=provider_inst)

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": provider_cls},
        run=run,
        writer=writer,
    ) as _:
        with patch(
            "coval_bench.runner.orchestrator._get_metrics",
            return_value=(_raising_wer, compute_rtf),
        ):
            summary = await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
            )

    by_metric = {r.metric_type: r for r in _recorded_rows(writer)}
    assert by_metric["TTFT"].status == ResultStatus.SUCCESS
    assert by_metric["WER"].status == ResultStatus.FAILED
    assert by_metric["WER"].metric_value is None
    assert "wer blew up" in (by_metric["WER"].error or "")
    assert summary.status == str(RunStatus.PARTIAL)


@pytest.mark.asyncio
async def test_tts_whisper_failure_emits_no_wer_row(settings: Settings) -> None:
    """A Whisper transcription failure (our instrument) is logged, emits no WER row, no PARTIAL."""

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "synth.wav"
        audio_path.write_bytes(b"\x00" * 512)

        hume_entry = _registry_entry(Benchmark.TTS, "hume")
        good_tts = TTSResult(
            provider="hume",
            model=hume_entry.model,
            voice=hume_entry.voice or "v",
            ttfa_ms=120.0,
            audio_path=audio_path,
            error=None,
            http_version="HTTP/2",
        )

        provider_inst = MagicMock()
        provider_inst.synthesize = AsyncMock(return_value=good_tts)
        provider_cls = MagicMock(return_value=provider_inst)

        matrix = [
            *_paused_registry(Benchmark.TTS),
            hume_entry.model_copy(update={"status": ModelStatus.ACTIVE}),
        ]

        run = _make_run()
        writer = _make_stub_writer(run)

        async with _orchestrator_env(
            audio_path=audio_path,
            tts_items=[_make_tts_item("hello world")],
            tts_providers={"hume": provider_cls},
            run=run,
            writer=writer,
        ) as _:
            with patch(
                "coval_bench.runner.orchestrator._transcribe_with_whisper",
                side_effect=RuntimeError("whisper down"),
            ):
                summary = await run_benchmarks(
                    settings=settings,
                    benchmark_kind="tts",
                    smoke=True,
                    matrix_overrides=matrix,
                )

        by_metric = {r.metric_type: r for r in _recorded_rows(writer)}
        assert by_metric["TTFA"].status == ResultStatus.SUCCESS
        assert "WER" not in by_metric  # instrument failure → no row, no run-status flip
        assert summary.status == str(RunStatus.SUCCEEDED)


@pytest.mark.asyncio
async def test_tts_wer_compute_failure_marked_failed(settings: Settings) -> None:
    """compute_wer crashing after a good Whisper transcript → FAILED WER row, run PARTIAL."""
    from coval_bench.metrics import compute_rtf

    def _raising_wer(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("wer blew up")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "synth.wav"
        audio_path.write_bytes(b"\x00" * 512)

        hume_entry = _registry_entry(Benchmark.TTS, "hume")
        good_tts = TTSResult(
            provider="hume",
            model=hume_entry.model,
            voice=hume_entry.voice or "v",
            ttfa_ms=120.0,
            audio_path=audio_path,
            error=None,
            http_version="HTTP/2",
        )

        provider_inst = MagicMock()
        provider_inst.synthesize = AsyncMock(return_value=good_tts)
        provider_cls = MagicMock(return_value=provider_inst)

        matrix = [
            *_paused_registry(Benchmark.TTS),
            hume_entry.model_copy(update={"status": ModelStatus.ACTIVE}),
        ]

        run = _make_run()
        writer = _make_stub_writer(run)

        async with _orchestrator_env(
            audio_path=audio_path,
            tts_items=[_make_tts_item("hello world")],
            tts_providers={"hume": provider_cls},
            run=run,
            writer=writer,
        ) as _:
            with (
                patch(
                    "coval_bench.runner.orchestrator._get_metrics",
                    return_value=(_raising_wer, compute_rtf),
                ),
                patch(
                    "coval_bench.runner.orchestrator._transcribe_with_whisper",
                    return_value="hello world",
                ),
            ):
                summary = await run_benchmarks(
                    settings=settings,
                    benchmark_kind="tts",
                    smoke=True,
                    matrix_overrides=matrix,
                )

        by_metric = {r.metric_type: r for r in _recorded_rows(writer)}
        assert by_metric["TTFA"].status == ResultStatus.SUCCESS
        assert by_metric["WER"].status == ResultStatus.FAILED
        assert by_metric["WER"].metric_value is None
        assert "wer blew up" in (by_metric["WER"].error or "")
        assert summary.status == str(RunStatus.PARTIAL)


# ---------------------------------------------------------------------------
# 18-22. per-item failure summary logging (silent paths log; no double-log)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_empty_result_logs_item_failure(audio_file: Path, settings: Settings) -> None:
    """A no-value return surfaces one stt_item_failed line keyed by metric."""
    empty = TranscriptionResult(provider="deepgram")  # all metrics None, no error

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=empty)

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": MagicMock(return_value=provider_inst)},
        run=run,
        writer=writer,
    ) as _:
        with structlog.testing.capture_logs() as captured:
            await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
            )

    failures = _events(captured, "stt_item_failed")
    assert len(failures) == 1, "one summary line per failed item, not one per row"
    event = failures[0]
    assert event["provider"] == "deepgram"
    assert event["item"] == "0001.wav"
    # Every silent FAILED metric is carried with its reason.
    assert {"TTFT", "AudioToFinal", "RTF"} <= set(event["reasons"])
    assert all("produced" in reason for reason in event["reasons"].values())


@pytest.mark.asyncio
async def test_stt_provider_error_logs_item_failure(audio_file: Path, settings: Settings) -> None:
    """A provider-set result.error (no raise) surfaces in the per-item summary."""
    errored = TranscriptionResult(
        provider="deepgram",
        ttft_seconds=0.3,
        complete_transcript="partial words",
        error="websocket closed unexpectedly",
    )

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=errored)

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": MagicMock(return_value=provider_inst)},
        run=run,
        writer=writer,
    ) as _:
        with structlog.testing.capture_logs() as captured:
            await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
            )

    failures = _events(captured, "stt_item_failed")
    assert len(failures) == 1
    reasons = failures[0]["reasons"]
    assert reasons  # the provider message reached the log
    assert all("websocket closed" in reason for reason in reasons.values())


@pytest.mark.asyncio
async def test_stt_provider_exception_not_double_logged(
    audio_file: Path, settings: Settings
) -> None:
    """A raised provider error warns once at its source — no second item summary."""
    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(side_effect=RuntimeError("boom"))

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": MagicMock(return_value=provider_inst)},
        run=run,
        writer=writer,
    ) as _:
        with structlog.testing.capture_logs() as captured:
            await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
            )

    assert _events(captured, "stt_provider_call_failed"), "source warning still emitted"
    assert not _events(captured, "stt_item_failed"), "exception failure must not be logged twice"


@pytest.mark.asyncio
async def test_stt_wer_crash_not_double_logged(audio_file: Path, settings: Settings) -> None:
    """A WER compute crash warns at its source; the item summary stays silent."""
    from coval_bench.metrics import compute_rtf

    def _raising_wer(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("wer blew up")

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=_good_transcription())

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],
        stt_providers={"deepgram": MagicMock(return_value=provider_inst)},
        run=run,
        writer=writer,
    ) as _:
        with (
            patch(
                "coval_bench.runner.orchestrator._get_metrics",
                return_value=(_raising_wer, compute_rtf),
            ),
            structlog.testing.capture_logs() as captured,
        ):
            await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
            )

    assert _events(captured, "wer_computation_failed"), "source warning still emitted"
    assert not _events(captured, "stt_item_failed"), "crash failure must not be logged twice"


@pytest.mark.asyncio
async def test_stt_ttfs_crash_not_double_logged(audio_file: Path, settings: Settings) -> None:
    """A TTFS compute crash warns at its source; the item summary stays silent."""

    def _raising_ttfs(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("ttfs blew up")

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=_good_transcription())

    run = _make_run()
    writer = _make_stub_writer(run)

    async with _orchestrator_env(
        audio_path=audio_file,
        stt_items=[_make_dataset_item(audio_file)],  # speech_end_offset_ms set → TTFS runs
        stt_providers={"deepgram": MagicMock(return_value=provider_inst)},
        run=run,
        writer=writer,
    ) as _:
        with (
            patch("coval_bench.metrics.compute_ttfs", _raising_ttfs),
            structlog.testing.capture_logs() as captured,
        ):
            await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=_only_stt_matrix("deepgram", "nova-2"),
            )

    assert _events(captured, "ttfs_computation_failed"), "source warning still emitted"
    assert not _events(captured, "stt_item_failed"), "crash failure must not be logged twice"


@pytest.mark.asyncio
async def test_tts_transport_gate_nulls_without_failing(settings: Settings) -> None:
    """An HTTP/1.1 contaminated TTFA is a null-valued success, not a logged item failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "synth.wav"
        audio_path.write_bytes(b"\x00" * 512)

        hume_entry = _registry_entry(Benchmark.TTS, "hume")
        tts_result = TTSResult(
            provider="hume",
            model=hume_entry.model,
            voice=hume_entry.voice or "v",
            ttfa_ms=120.0,
            audio_path=audio_path,
            error=None,
            http_version="HTTP/1.1",
        )

        provider_inst = MagicMock()
        provider_inst.synthesize = AsyncMock(return_value=tts_result)
        provider_cls = MagicMock(return_value=provider_inst)

        matrix = [
            *_paused_registry(Benchmark.TTS),
            hume_entry.model_copy(update={"status": ModelStatus.ACTIVE}),
        ]

        run = _make_run()
        writer = _make_stub_writer(run)

        async with _orchestrator_env(
            audio_path=audio_path,
            tts_items=[_make_tts_item("hello world")],
            tts_providers={"hume": provider_cls},
            run=run,
            writer=writer,
        ) as _:
            with (
                patch(
                    "coval_bench.runner.orchestrator._transcribe_with_whisper",
                    return_value="hello world",
                ),
                structlog.testing.capture_logs() as captured,
            ):
                await run_benchmarks(
                    settings=settings,
                    benchmark_kind="tts",
                    smoke=True,
                    matrix_overrides=matrix,
                )

    assert _events(captured, "tts_item_failed") == []

    ttfa = {r.metric_type: r for r in _recorded_rows(writer)}["TTFA"]
    assert ttfa.status == ResultStatus.SUCCESS
    assert ttfa.metric_value is None
    assert "HTTP/1.1" in (ttfa.error or "")


# ---------------------------------------------------------------------------
# 11. PostHog run events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_posthog_completed_event(audio_file: Path, settings: Settings) -> None:
    """A successful run emits 'benchmark run completed' and flushes."""
    fake = create_autospec(Posthog, instance=True)
    settings = settings.model_copy(
        update={"posthog_project_token": "phc_test", "posthog_disabled": False}
    )

    provider = MagicMock()
    provider.measure_ttft = AsyncMock(return_value=_good_transcription())
    provider_cls = MagicMock(return_value=provider)
    matrix = [_stt_entry("deepgram", "nova-2")]

    run = _make_run()
    writer = _make_stub_writer(run)

    with patch("coval_bench.runner.orchestrator.Posthog", lambda *a, **k: fake):
        async with _orchestrator_env(
            audio_path=audio_file,
            stt_items=[_make_dataset_item(audio_file)],
            stt_providers={"deepgram": provider_cls},
            run=run,
            writer=writer,
        ):
            summary = await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=matrix,
            )

    assert summary.status == str(RunStatus.SUCCEEDED)
    fake.capture.assert_called_once()
    assert fake.capture.call_args.args[0] == "benchmark_run_completed"
    assert fake.capture.call_args.kwargs["distinct_id"] == "coval-bench-runner"
    properties = fake.capture.call_args.kwargs["properties"]
    assert properties["status"] == str(RunStatus.SUCCEEDED)
    assert properties["$process_person_profile"] is False
    fake.flush.assert_called_once()


@pytest.mark.asyncio
async def test_posthog_failed_event(settings: Settings) -> None:
    """An unrecoverable run error emits 'benchmark run failed', flushes, re-raises."""
    fake = create_autospec(Posthog, instance=True)
    settings = settings.model_copy(
        update={"posthog_project_token": "phc_test", "posthog_disabled": False}
    )

    class _DatasetIntegrityError(Exception):
        pass

    def _bad_load(dataset_id: str, *, settings: Any, sample_size: int | None = None) -> Any:
        raise _DatasetIntegrityError("hash mismatch")

    deepgram_cls = MagicMock()
    deepgram_cls.warmup = AsyncMock(return_value=None)
    run = _make_run()
    writer = _make_stub_writer(run)
    fake_pool = MagicMock()

    @contextlib.asynccontextmanager
    async def _fake_lifespan_pool(s: Any) -> AsyncIterator[MagicMock]:
        yield fake_pool

    models_mod = MagicMock()
    models_mod.Benchmark = Benchmark
    models_mod.Result = Result
    models_mod.ResultStatus = ResultStatus
    models_mod.RunStatus = RunStatus

    def _fake_get_db_symbols() -> tuple[Any, Any, Any, Any]:
        return _fake_lifespan_pool, MagicMock(return_value=writer), RunStatus, models_mod

    with (
        patch("coval_bench.runner.orchestrator.Posthog", lambda *a, **k: fake),
        patch(
            "coval_bench.runner.orchestrator._get_db_symbols",
            side_effect=_fake_get_db_symbols,
        ),
        patch(
            "coval_bench.runner.orchestrator._get_stt_providers",
            return_value={"deepgram": deepgram_cls},
        ),
        patch("coval_bench.runner.orchestrator._get_tts_providers", return_value={}),
        patch("coval_bench.runner.orchestrator._get_load_dataset", return_value=_bad_load),
        patch(
            "coval_bench.runner.orchestrator._get_metrics",
            return_value=(MagicMock(), MagicMock()),
        ),
        pytest.raises(_DatasetIntegrityError),
    ):
        await run_benchmarks(
            settings=settings,
            benchmark_kind="stt",
            matrix_overrides=[_stt_entry("deepgram", "nova-2")],
        )

    fake.capture.assert_called_once()
    assert fake.capture.call_args.args[0] == "benchmark_run_failed"
    assert fake.capture.call_args.kwargs["properties"]["$process_person_profile"] is False
    fake.flush.assert_called_once()


@pytest.mark.asyncio
async def test_posthog_capture_failure_does_not_fail_run(
    audio_file: Path, settings: Settings
) -> None:
    """A raising PostHog client must not flip a successful run to FAILED."""
    fake = create_autospec(Posthog, instance=True)
    fake.capture.side_effect = RuntimeError("posthog down")
    settings = settings.model_copy(
        update={"posthog_project_token": "phc_test", "posthog_disabled": False}
    )

    provider = MagicMock()
    provider.measure_ttft = AsyncMock(return_value=_good_transcription())
    provider_cls = MagicMock(return_value=provider)
    matrix = [_stt_entry("deepgram", "nova-2")]

    run = _make_run()
    writer = _make_stub_writer(run)

    with patch("coval_bench.runner.orchestrator.Posthog", lambda *a, **k: fake):
        async with _orchestrator_env(
            audio_path=audio_file,
            stt_items=[_make_dataset_item(audio_file)],
            stt_providers={"deepgram": provider_cls},
            run=run,
            writer=writer,
        ):
            summary = await run_benchmarks(
                settings=settings,
                benchmark_kind="stt",
                smoke=True,
                matrix_overrides=matrix,
            )

    assert summary.status == str(RunStatus.SUCCEEDED)
    fake.capture.assert_called_once()
