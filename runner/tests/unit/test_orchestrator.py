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
10. test_disabled_providers_skipped — enabled=False entries not instantiated
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.db.models import Benchmark, Result, ResultStatus, Run, RunStatus
from coval_bench.providers.base import TranscriptionResult, TTSResult
from coval_bench.runner.config import ProviderEntry
from coval_bench.runner.orchestrator import RunSummary, run_benchmarks
from coval_bench.runner.retry import with_retry

# ---------------------------------------------------------------------------
# Shared test settings
# ---------------------------------------------------------------------------

_TEST_SETTINGS = Settings(
    database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
    dataset_bucket="test-bucket",
    dataset_id="stt-v1",
    runner_sha="test-sha",
    log_level="DEBUG",
    openai_api_key=SecretStr("sk-test"),
    deepgram_api_key=SecretStr("dg-test"),
)


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------


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
    item.metadata = {}
    return item


def _make_tts_item(transcript: str = "hello world") -> Any:
    item = MagicMock()
    item.transcript = transcript
    return item


def _good_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        provider="deepgram",
        ttft_seconds=0.25,
        total_time=0.9,
        audio_to_final_seconds=0.85,
        rtf_value=0.85,
        first_token_content="hello",  # noqa: S106 — not a password
        complete_transcript="hello world",
    )


def _make_stub_writer(run: Run) -> MagicMock:
    writer = MagicMock()
    writer.start_run = AsyncMock(return_value=run)
    writer.record_results = AsyncMock()
    writer.finish_run = AsyncMock()
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

    stt_dataset = MagicMock()
    stt_dataset.items = stt_items
    tts_dataset = MagicMock()
    tts_dataset.items = tts_items

    def _load_dataset(dataset_id: str, *, settings: Any) -> Any:
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
    """Tiny placeholder file (enough for read_bytes())."""
    f = tmp_path / "0001.wav"
    f.write_bytes(b"\x00" * 1024)
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
        ProviderEntry(provider="deepgram", model="nova-2", enabled=True),
        ProviderEntry(provider="elevenlabs", model="scribe_v2_realtime", enabled=True),
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
        ProviderEntry(provider="deepgram", model="nova-2", enabled=True),
        ProviderEntry(provider="elevenlabs", model="scribe_v2_realtime", enabled=True),
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
    # Override the full DEFAULT_STT_MATRIX so only nova-2 deepgram runs
    from coval_bench.runner.config import DEFAULT_STT_MATRIX

    matrix = [
        # Disable all defaults and enable only the one failing provider
        *[
            ProviderEntry(provider=e.provider, model=e.model, voice=e.voice, enabled=False)
            for e in DEFAULT_STT_MATRIX
        ],
        ProviderEntry(provider="deepgram", model="nova-2", enabled=True),
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
    assert summary.fail_count == 1
    assert summary.success_count == 0


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
    matrix = [ProviderEntry(provider="deepgram", model="nova-2", enabled=True)]

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

    def _bad_load(dataset_id: str, *, settings: Any) -> Any:
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

    with (
        patch(
            "coval_bench.runner.orchestrator._get_db_symbols",
            side_effect=_fake_get_db_symbols,
        ),
        patch(
            "coval_bench.runner.orchestrator._get_stt_providers",
            return_value={"deepgram": MagicMock()},
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
            matrix_overrides=[ProviderEntry(provider="deepgram", model="nova-2", enabled=True)],
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

        tts_providers = {"elevenlabs": provider_cls}
        tts_item = _make_tts_item("hello world")

        run = _make_run()
        writer = _make_stub_writer(run)

        tts_dataset = MagicMock()
        tts_dataset.items = [tts_item]

        def _load(dataset_id: str, *, settings: Any) -> Any:
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
            ProviderEntry(
                provider="elevenlabs",
                model="eleven_flash_v2_5",
                voice="IKne3meq5aSn9XLyUdCD",
                enabled=True,
            )
        ]

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

    # Override: nova-2 enabled, nova-3 disabled; disable all others from DEFAULT too
    from coval_bench.runner.config import DEFAULT_STT_MATRIX

    full_matrix = [
        ProviderEntry(provider=e.provider, model=e.model, voice=e.voice, enabled=False)
        for e in DEFAULT_STT_MATRIX
    ]
    # Flip only nova-2 on
    full_matrix_map = {(e.provider, e.model): e for e in full_matrix}
    full_matrix_map[("deepgram", "nova-2")] = ProviderEntry(
        provider="deepgram", model="nova-2", enabled=True
    )
    full_matrix_map[("deepgram", "nova-3")] = ProviderEntry(
        provider="deepgram", model="nova-3", enabled=False
    )
    matrix = list(full_matrix_map.values())

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


# ---------------------------------------------------------------------------
# 10. test_disabled_providers_skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_providers_skipped(audio_file: Path, settings: Settings) -> None:
    """Entries with enabled=False are never instantiated or called."""
    good = _good_transcription()

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=good)
    provider_cls = MagicMock(return_value=provider_inst)

    disabled_cls = MagicMock()

    stt_providers = {"deepgram": provider_cls, "elevenlabs": disabled_cls}
    matrix = [
        ProviderEntry(provider="deepgram", model="nova-2", enabled=True),
        ProviderEntry(provider="elevenlabs", model="scribe_v2_realtime", enabled=False),
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
# 11. test_disabled_flag_skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_flag_skipped(audio_file: Path, settings: Settings) -> None:
    """Entries with disabled=True are never instantiated or called, even when enabled=True."""
    good = _good_transcription()

    provider_inst = MagicMock()
    provider_inst.measure_ttft = AsyncMock(return_value=good)
    provider_cls = MagicMock(return_value=provider_inst)

    disabled_cls = MagicMock()

    stt_providers = {"deepgram": provider_cls, "elevenlabs": disabled_cls}
    matrix = [
        ProviderEntry(provider="deepgram", model="nova-2", enabled=True),
        # enabled=True but disabled=True — the disabled flag must win
        ProviderEntry(
            provider="elevenlabs", model="scribe_v2_realtime", enabled=True, disabled=True
        ),
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

    # The disabled provider class must never have been instantiated or called
    disabled_cls.assert_not_called()
    # The active provider was still called
    provider_cls.assert_called()


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
        ProviderEntry(provider="deepgram", model="nova-2", enabled=True),
        ProviderEntry(provider="elevenlabs", model="scribe_v2_realtime", enabled=True),
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
async def test_sigterm_finalizes_run_as_partial(
    audio_file: Path, settings: Settings
) -> None:
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

    matrix = [ProviderEntry(provider="deepgram", model="nova-2", enabled=True)]

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
