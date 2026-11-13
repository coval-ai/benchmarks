# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the latency probe (coval_bench.runner.latency_probe).

Providers, the registry and the database write are faked, so nothing leaves the
process.
"""

from __future__ import annotations

import asyncio
import wave
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult, TTSResult
from coval_bench.registries import METRIC_EXCLUSIONS, Benchmark, Metric, RegisteredModel
from coval_bench.registries.models import Voice
from coval_bench.runner import latency_probe

_TICK = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
_SETTINGS = cast(Settings, SimpleNamespace(schedule_period_seconds=1800))


@asynccontextmanager
async def _free_lock(_pool: Any, _key: str, *, timeout_s: float) -> Any:
    yield True


def _model(benchmark: Benchmark, **fields: Any) -> RegisteredModel:
    return RegisteredModel(
        benchmark=benchmark,
        provider="acme",
        model="fast-1",
        collected=True,
        published=False,
        **fields,
    )


def _tts_result(**fields: Any) -> TTSResult:
    defaults: dict[str, Any] = {
        "provider": "acme",
        "model": "fast-1",
        "voice": "v",
        "ttfa_ms": 180.0,
        "audio_path": None,
        "error": None,
        "http_version": "HTTP/2",
        "connection_reused": True,
    }
    return TTSResult(**{**defaults, **fields})


def _tts_provider(outcome: Any, calls: list[dict[str, Any]]) -> Any:
    class Provider:
        def __init__(self, *, settings: Any, model: str, voice: str | None) -> None:
            calls.append({"model": model, "voice": voice})

        async def synthesize(self, text: str) -> TTSResult:
            calls[-1]["text"] = text
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return cast(TTSResult, await outcome())
            return cast(TTSResult, outcome)

    return Provider


async def _probe_tts(
    monkeypatch: pytest.MonkeyPatch, outcome: Any, **model_fields: Any
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        latency_probe, "_get_tts_providers", lambda: {"acme": _tts_provider(outcome, calls)}
    )
    result = await latency_probe._probe_tts(
        _model(Benchmark.TTS, **model_fields),
        model_id=1,
        settings=_SETTINGS,
        scheduled_at=_TICK,
    )
    return result, calls


async def test_tts_probe_stores_ttfa_for_the_probe_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voices = (Voice(id="f-1", gender="female"), Voice(id="m-1", gender="male"))
    result, calls = await _probe_tts(monkeypatch, _tts_result(), voice="fallback", voices=voices)

    assert calls == [{"model": "fast-1", "voice": "f-1", "text": latency_probe.TTS_PROBE_TEXT}]
    assert (result.status, result.ttfa_ms, result.voice) == ("succeeded", 180.0, "f-1")
    assert result.scheduled_at == _TICK


async def test_tts_probe_records_a_provider_error_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = await _probe_tts(monkeypatch, RuntimeError("503 from upstream"))

    assert len(calls) == 1
    assert (result.status, result.failure_origin) == ("failed", "provider")
    assert result.error == "503 from upstream"
    assert result.ttfa_ms is None


async def test_tts_probe_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latency_probe, "_TTS_TIMEOUT_S", 0.01)

    async def hang() -> TTSResult:
        await asyncio.sleep(5)
        return _tts_result()

    result, _ = await _probe_tts(monkeypatch, hang)

    assert (result.status, result.failure_origin) == ("failed", "provider")
    assert result.error == "timed out after 0.01s"


async def test_tts_probe_on_a_cold_connection_is_our_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = await _probe_tts(monkeypatch, _tts_result(connection_reused=False))

    assert (result.status, result.failure_origin) == ("failed", "runner")
    assert result.ttfa_ms is None


async def test_tts_probe_deletes_the_synthesized_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "out.wav"
    audio.write_bytes(b"audio")

    await _probe_tts(monkeypatch, _tts_result(audio_path=audio))

    assert not audio.exists()


async def test_unknown_provider_is_our_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latency_probe, "_get_tts_providers", dict)

    result = await latency_probe._probe_tts(
        _model(Benchmark.TTS), model_id=1, settings=_SETTINGS, scheduled_at=_TICK
    )

    assert (result.status, result.failure_origin) == ("failed", "runner")


def _stt_item(tmp_path: Path) -> Any:
    path = tmp_path / "hello.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    return SimpleNamespace(path=path, duration_sec=1.0, speech_end_offset_ms=600.0)


async def _probe_stt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: Any) -> Any:
    class Provider:
        def __init__(self, settings: Any, model: str) -> None:
            pass

        async def measure_ttft(self, *args: Any) -> TranscriptionResult:
            if isinstance(outcome, BaseException):
                raise outcome
            return cast(TranscriptionResult, outcome)

    monkeypatch.setattr(latency_probe, "_get_stt_providers", lambda: {"acme": Provider})
    return await latency_probe._probe_stt(
        _model(Benchmark.STT),
        _stt_item(tmp_path),
        model_id=1,
        settings=_SETTINGS,
        scheduled_at=_TICK,
    )


async def test_stt_probe_stores_ttft_and_ttfs_in_ms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcription = TranscriptionResult(
        provider="acme",
        ttft_seconds=0.25,
        audio_to_final_seconds=0.9,
        complete_transcript="hello world",
    )

    result = await _probe_stt(monkeypatch, tmp_path, transcription)

    assert result.status == "succeeded"
    assert result.ttft_ms == pytest.approx(250.0)
    assert result.ttfs_ms == pytest.approx(300.0)


async def test_stt_probe_without_a_final_is_a_failure_that_keeps_ttft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcription = TranscriptionResult(
        provider="acme", ttft_seconds=0.25, partial_transcripts=["hello"]
    )

    result = await _probe_stt(monkeypatch, tmp_path, transcription)

    assert (result.status, result.failure_origin) == ("failed", "provider")
    assert result.error == "no TTFS produced"
    assert result.ttft_ms == pytest.approx(250.0)
    assert result.ttfs_ms is None


async def test_stt_probe_skips_a_metric_the_model_is_excluded_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(METRIC_EXCLUSIONS, Metric.TTFT, {("acme", "fast-1")})
    transcription = TranscriptionResult(
        provider="acme", audio_to_final_seconds=0.9, complete_transcript="hello world"
    )

    result = await _probe_stt(monkeypatch, tmp_path, transcription)

    assert result.status == "succeeded"
    assert result.ttft_ms is None
    assert result.ttfs_ms == pytest.approx(300.0)


async def test_stt_probe_records_a_reported_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = await _probe_stt(
        monkeypatch, tmp_path, TranscriptionResult(provider="acme", error="auth failed")
    )

    assert (result.status, result.failure_origin, result.error) == (
        "failed",
        "provider",
        "auth failed",
    )
    assert result.ttft_ms is None


async def test_run_probes_every_collected_model_and_stores_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster = [
        _model(Benchmark.TTS),
        RegisteredModel(
            benchmark=Benchmark.TTS, provider="acme", model="off", collected=False, published=False
        ),
        RegisteredModel(
            benchmark=Benchmark.TTS, provider="down", model="tts-2", collected=True, published=True
        ),
        _model(Benchmark.STT),
    ]
    stored: list[Any] = []
    closed: list[bool] = []

    @asynccontextmanager
    async def fake_pool(_settings: Any) -> Any:
        yield object()

    async def fake_fetch(_pool: Any) -> list[tuple[int, RegisteredModel]]:
        return list(enumerate(roster, start=1))

    async def fake_insert(_pool: Any, result: Any) -> bool:
        stored.append(result)
        return True

    async def fake_close() -> None:
        closed.append(True)

    async def no_warmup(*_: Any) -> None:
        return None

    monkeypatch.setattr(latency_probe, "lifespan_pool", fake_pool)
    monkeypatch.setattr(latency_probe, "fetch_models_with_ids", fake_fetch)
    monkeypatch.setattr(latency_probe, "insert_probe_result", fake_insert)
    monkeypatch.setattr(latency_probe, "model_lock", _free_lock)
    monkeypatch.setattr(latency_probe, "_close_http_clients", fake_close)
    monkeypatch.setattr(latency_probe, "_warmup", no_warmup)
    monkeypatch.setattr(latency_probe, "_current_tick", lambda _settings: _TICK)
    monkeypatch.setattr(
        latency_probe,
        "_get_tts_providers",
        lambda: {
            "acme": _tts_provider(_tts_result(), []),
            "down": _tts_provider(RuntimeError("down"), []),
        },
    )

    summary = await latency_probe.run_latency_probes(settings=_SETTINGS, kind="tts")

    assert (summary.scheduled_at, summary.probed, summary.stored) == (_TICK, 2, 2)
    assert (summary.failed, summary.skipped, summary.errored) == (1, 0, 0)
    assert {(r.model_id, r.status) for r in stored} == {(1, "succeeded"), (3, "failed")}
    assert closed == [True]


def _wire_run(monkeypatch: pytest.MonkeyPatch, insert: Any) -> None:
    roster = [
        _model(Benchmark.TTS),
        RegisteredModel(
            benchmark=Benchmark.TTS, provider="acme", model="fast-2", collected=True, published=True
        ),
    ]

    @asynccontextmanager
    async def fake_pool(_settings: Any) -> Any:
        yield object()

    async def fake_fetch(_pool: Any) -> list[tuple[int, RegisteredModel]]:
        return list(enumerate(roster, start=1))

    async def nothing(*_: Any) -> None:
        return None

    monkeypatch.setattr(latency_probe, "lifespan_pool", fake_pool)
    monkeypatch.setattr(latency_probe, "fetch_models_with_ids", fake_fetch)
    monkeypatch.setattr(latency_probe, "insert_probe_result", insert)
    monkeypatch.setattr(latency_probe, "model_lock", _free_lock)
    monkeypatch.setattr(latency_probe, "_close_http_clients", nothing)
    monkeypatch.setattr(latency_probe, "_warmup", nothing)
    monkeypatch.setattr(latency_probe, "_current_tick", lambda _settings: _TICK)
    monkeypatch.setattr(
        latency_probe, "_get_tts_providers", lambda: {"acme": _tts_provider(_tts_result(), [])}
    )


async def test_one_failed_write_does_not_stop_the_other_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: list[int] = []

    async def flaky_insert(_pool: Any, result: Any) -> bool:
        if result.model_id == 1:
            raise RuntimeError("permission denied for table probe_results")
        stored.append(result.model_id)
        return True

    _wire_run(monkeypatch, flaky_insert)

    summary = await latency_probe.run_latency_probes(settings=_SETTINGS, kind="tts")

    assert stored == [2]
    assert (summary.probed, summary.stored, summary.errored) == (2, 1, 1)


async def test_a_probe_the_tick_already_has_is_counted_as_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def already_there(_pool: Any, _result: Any) -> bool:
        return False

    _wire_run(monkeypatch, already_there)

    summary = await latency_probe.run_latency_probes(settings=_SETTINGS, kind="tts")

    assert (summary.probed, summary.stored, summary.skipped) == (2, 0, 2)


async def test_a_model_another_job_holds_is_skipped_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    stored: list[int] = []

    async def insert(_pool: Any, result: Any) -> bool:
        stored.append(result.model_id)
        return True

    @asynccontextmanager
    async def fast_1_is_busy(_pool: Any, key: str, *, timeout_s: float) -> Any:
        yield key != "model_lock:acme/fast-1"

    _wire_run(monkeypatch, insert)
    monkeypatch.setattr(latency_probe, "model_lock", fast_1_is_busy)
    monkeypatch.setattr(
        latency_probe, "_get_tts_providers", lambda: {"acme": _tts_provider(_tts_result(), calls)}
    )

    summary = await latency_probe.run_latency_probes(settings=_SETTINGS, kind="tts")

    assert len(calls) == 1
    assert stored == [2]
    assert (summary.probed, summary.stored, summary.skipped) == (2, 1, 1)


async def test_models_are_probed_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    running = 0
    peak = 0

    async def slow() -> TTSResult:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return _tts_result()

    async def insert(_pool: Any, _result: Any) -> bool:
        return True

    _wire_run(monkeypatch, insert)
    monkeypatch.setattr(
        latency_probe, "_get_tts_providers", lambda: {"acme": _tts_provider(slow, [])}
    )

    await latency_probe.run_latency_probes(settings=_SETTINGS, kind="tts")

    assert peak == 1


@pytest.mark.parametrize(
    "error",
    [
        "server rejected WebSocket connection: HTTP 401",
        "task_failed: [1008] insufficient balance",
        "429 Too Many Requests",
    ],
)
async def test_a_key_problem_is_our_failure(monkeypatch: pytest.MonkeyPatch, error: str) -> None:
    result, _ = await _probe_tts(monkeypatch, RuntimeError(error))

    assert (result.status, result.failure_origin, result.error) == ("failed", "runner", error)


async def test_a_reported_401_is_our_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = _tts_result(ttfa_ms=None, error="request failed", status_code=401)

    result, _ = await _probe_tts(monkeypatch, outcome)

    assert (result.status, result.failure_origin, result.status_code) == ("failed", "runner", 401)


async def test_a_providers_own_timeout_is_not_reported_as_ours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = await _probe_tts(monkeypatch, TimeoutError("timed out during opening handshake"))

    assert result.error == "timed out during opening handshake"
    assert result.failure_origin == "provider"
