# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.guava (GuavaSTTProvider).

All tests use FakeWebSocket — no live network calls.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt.guava import GuavaSTTProvider, ws_url_from_base
from tests.providers.stt.conftest import FakeWebSocket

_BASE_URL = "https://guava.example.internal"

# A few frames of PCM — enough to exercise pacing without the 3 s fixture
# dominating test runtime.
_SMALL_PCM = b"\x01\x02" * 1100


_DOMAIN = "test-domain"


def make_provider(api_key: SecretStr | None = None) -> GuavaSTTProvider:
    return GuavaSTTProvider(
        api_key=api_key or SecretStr("test-key"), base_url=_BASE_URL, domain=_DOMAIN
    )


def _fake_connect(events: list[Any], sent: list[Any] | None = None) -> Any:
    ws = FakeWebSocket(events, on_send=None if sent is None else sent.append)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_guava_success(fake_api_key: SecretStr) -> None:
    events: list[Any] = [
        {"type": "asr_partial", "transcript": "hello", "ts_ms": 100},
        {"type": "asr_partial", "transcript": "hello world", "ts_ms": 700},
        {"type": "asr_final", "transcript": "hello world", "ts_ms": 900},
    ]
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.guava.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.ttft_seconds is not None and result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2
    assert result.audio_to_final_seconds is not None
    assert result.partial_transcripts == ["hello", "hello world"]
    wer = compute_wer("hello world", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_guava_uses_final_over_partial(fake_api_key: SecretStr) -> None:
    events: list[Any] = [
        {"type": "asr_partial", "transcript": "hello there", "ts_ms": 300},
        {"type": "asr_final", "transcript": "hello their", "ts_ms": 900},
    ]
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.guava.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "hello their"


@pytest.mark.asyncio
async def test_guava_final_timing_is_not_replaced_by_partial() -> None:
    result = TranscriptionResult(provider="guava", audio_start_time=100.0)
    ws = FakeWebSocket(
        [
            {"type": "asr_partial", "transcript": "early guess"},
            {"type": "asr_final", "transcript": "committed result"},
        ]
    )
    with patch("coval_bench.providers.stt.guava.time.monotonic", side_effect=[101.0, 103.0]):
        await make_provider()._receive(ws, result)
    assert result.ttft_seconds == 1.0
    assert result.audio_to_final_seconds == 3.0
    assert result.complete_transcript == "committed result"


@pytest.mark.asyncio
async def test_guava_partial_only_close_is_failure() -> None:
    with patch(
        "coval_bench.providers.stt.guava.ws_client.connect",
        return_value=_fake_connect([{"type": "asr_partial", "transcript": "uncommitted"}]),
    ):
        result = await make_provider().measure_ttft(_SMALL_PCM, 1, 2, 16000)
    assert result.error is not None
    assert "before a final transcription" in result.error
    assert result.complete_transcript is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_guava_accumulates_multi_segment_finals(fake_api_key: SecretStr) -> None:
    """Multiple finals concatenate in arrival order."""
    events: list[Any] = [
        {"type": "asr_final", "transcript": "hello", "ts_ms": 400},
        {"type": "asr_final", "transcript": "world", "ts_ms": 900},
    ]
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.guava.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2


@pytest.mark.asyncio
async def test_guava_sends_start_binary_and_eos(fake_api_key: SecretStr) -> None:
    """Opens with a start frame, streams PCM as binary, ends with eos."""
    events: list[Any] = [{"type": "asr_final", "transcript": "ok", "ts_ms": 500}]
    sent: list[Any] = []
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.guava.ws_client.connect",
        return_value=_fake_connect(events, sent),
    ):
        await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    start = json.loads(sent[0])
    assert start == {"type": "start", "domain": _DOMAIN, "partial_transcripts": True}
    binary = [msg for msg in sent if isinstance(msg, (bytes, bytearray))]
    assert b"".join(binary) == _SMALL_PCM
    assert json.loads(sent[-1]) == {"type": "eos"}


@pytest.mark.asyncio
async def test_guava_sends_bearer_when_key_present(fake_api_key: SecretStr) -> None:
    headers: dict[str, Any] = {}

    def _connect(url: str, **kwargs: Any) -> Any:
        headers.update({"additional_headers": kwargs.get("additional_headers")})
        return _fake_connect([{"type": "asr_final", "transcript": "ok", "ts_ms": 1}])

    provider = make_provider(fake_api_key)
    with patch("coval_bench.providers.stt.guava.ws_client.connect", side_effect=_connect):
        await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert headers["additional_headers"] == {
        "Authorization": f"Bearer {fake_api_key.get_secret_value()}"
    }


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="guava_api_key is required"):
        GuavaSTTProvider(api_key=None, base_url=_BASE_URL)


def test_blank_api_key_raises() -> None:
    with pytest.raises(ValueError, match="guava_api_key is required"):
        GuavaSTTProvider(api_key=SecretStr("   "), base_url=_BASE_URL)


def test_provider_name() -> None:
    assert make_provider().name == "guava"


def test_provider_model() -> None:
    assert make_provider().model == "daytona-stt"


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Guava STT model"):
        GuavaSTTProvider(api_key=SecretStr("k"), model="whisper", base_url=_BASE_URL)


def test_missing_base_url_raises() -> None:
    with pytest.raises(ValueError, match="guava_base_url is required"):
        GuavaSTTProvider(api_key=SecretStr("k"), base_url=None, domain=_DOMAIN)


def test_missing_domain_raises() -> None:
    with pytest.raises(ValueError, match="guava_stt_domain is required"):
        GuavaSTTProvider(api_key=SecretStr("k"), base_url=_BASE_URL, domain=None)


def test_ws_url_from_base() -> None:
    assert ws_url_from_base("http://localhost:8000") == "ws://localhost:8000/audio/transcriptions"
    assert ws_url_from_base("https://guava.example.internal/") == (
        "wss://guava.example.internal/audio/transcriptions"
    )


@pytest.mark.asyncio
async def test_guava_wrong_sample_rate(fake_api_key: SecretStr) -> None:
    provider = make_provider(fake_api_key)
    result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 8000)
    assert result.error is not None
    assert "16 kHz" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_guava_rejects_non_mono(fake_api_key: SecretStr) -> None:
    provider = make_provider(fake_api_key)
    result = await provider.measure_ttft(_SMALL_PCM, 2, 2, 16000)
    assert result.error is not None
    assert "mono 16-bit" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_guava_error_event(fake_api_key: SecretStr) -> None:
    events: list[Any] = [{"type": "error", "message": "Rate limit exceeded"}]
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.guava.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "Rate limit exceeded" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_guava_stream_ends_without_final(fake_api_key: SecretStr) -> None:
    """A stream that closes with no final surfaces an error, not a silent pass."""
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.guava.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "before a final transcription" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_guava_connection_error(fake_api_key: SecretStr) -> None:
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.guava.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.complete_transcript is None


def test_guava_smoke_passes_configured_domain(tmp_path: Path) -> None:
    from coval_bench.__main__ import cli
    from coval_bench.config import Settings

    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(_SMALL_PCM)
    settings = Settings(
        _env_file=None,
        guava_api_key=SecretStr("test-key"),
        guava_base_url=_BASE_URL,
        guava_stt_domain=_DOMAIN,
    )
    measured = AsyncMock(
        return_value=TranscriptionResult(provider="guava", complete_transcript="ok")
    )
    with (
        patch("coval_bench.config.get_settings", return_value=settings),
        patch.object(GuavaSTTProvider, "measure_ttft", measured),
    ):
        result = CliRunner().invoke(
            cli, ["stt-smoke", "--provider", "guava", "--model", "daytona-stt", "--wav", str(audio)]
        )
    assert result.exit_code == 0, result.output
    measured.assert_awaited_once()
