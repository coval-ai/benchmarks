# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.modulate (ModulateSTTProvider).

All tests use FakeWebSocket — no live network calls. The event fixtures mirror
the Modulate streaming wire protocol: ``partial_utterance`` previews, committed
``utterance`` finals, and a terminating ``done`` message. The transcript is the
concatenation of every ``utterance.text`` in arrival order.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.modulate import ModulateSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events

_MULTILINGUAL = "multilingual-transcription-streaming"


def make_provider(model: str = "english-fast-transcription-streaming") -> ModulateSTTProvider:
    return ModulateSTTProvider(api_key=SecretStr("test-key-modulate"), model=model)


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path — final utterance text forms the transcript
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_modulate_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("modulate", "events-success")
    provider = ModulateSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.modulate.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_modulate_ttft_fires_on_first_partial(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """TTFT is time-to-first-word: it fires on a partial, before any final."""
    events: list[Any] = [
        {"type": "partial_utterance", "partial_utterance": {"text": "hello"}},
        {"type": "utterance", "utterance": {"text": "hello world"}},
        {"type": "done", "duration_ms": 2000},
    ]
    provider = ModulateSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.modulate.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.first_token_content == "hello"  # noqa: S105 - transcript fixture text


@pytest.mark.asyncio
async def test_modulate_excludes_partials_from_transcript(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Only ``utterance`` messages form the transcript; partials are dropped."""
    events: list[Any] = [
        {"type": "partial_utterance", "partial_utterance": {"text": "wrong guess entirely"}},
        {"type": "utterance", "utterance": {"text": "hello world"}},
        {"type": "done", "duration_ms": 2000},
    ]
    provider = ModulateSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.modulate.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2


@pytest.mark.asyncio
async def test_modulate_multilingual_concatenates_utterances(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Multiple per-utterance finals are concatenated in arrival order."""
    events: list[Any] = [
        {"type": "partial_utterance", "partial_utterance": {"text": "bonjour"}},
        {"type": "utterance", "utterance": {"text": "bonjour le monde", "speaker": 1}},
        {"type": "utterance", "utterance": {"text": "comment ca va", "speaker": 1}},
        {"type": "done", "duration_ms": 4000},
    ]
    provider = ModulateSTTProvider(api_key=fake_api_key, model=_MULTILINGUAL)

    with patch(
        "coval_bench.providers.stt.modulate.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.complete_transcript == "bonjour le monde comment ca va"


@pytest.mark.asyncio
async def test_modulate_sends_empty_eos_frame(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """End-of-audio is signalled with an empty text frame after the PCM chunks."""
    ws = FakeWebSocket([{"type": "done", "duration_ms": 100}])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    provider = ModulateSTTProvider(api_key=fake_api_key)
    with patch("coval_bench.providers.stt.modulate.ws_client.connect", return_value=cm):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert ws._sent[-1] == ""


@pytest.mark.asyncio
async def test_modulate_multilingual_url_requests_partials(fake_api_key: SecretStr) -> None:
    """The multilingual endpoint asks for partials and disables diarization."""
    provider = ModulateSTTProvider(api_key=fake_api_key, model=_MULTILINGUAL)
    url = provider._build_websocket_url(16000)
    assert "velma-2-stt-streaming?" in url
    assert "partial_results=true" in url
    assert "speaker_diarization=false" in url


def test_modulate_english_url_omits_partials(fake_api_key: SecretStr) -> None:
    provider = ModulateSTTProvider(api_key=fake_api_key)
    url = provider._build_websocket_url(16000)
    assert "velma-2-stt-streaming-english-v2?" in url
    assert "partial_results" not in url


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "modulate-english-fast-transcription-streaming"
    assert make_provider(_MULTILINGUAL).name == f"modulate-{_MULTILINGUAL}"


def test_provider_model() -> None:
    assert make_provider().model == "english-fast-transcription-streaming"


# ---------------------------------------------------------------------------
# Invalid construction
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Modulate model"):
        ModulateSTTProvider(api_key=SecretStr("k"), model="velma-3")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="modulate_api_key is required"):
        ModulateSTTProvider(api_key=None)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_modulate_rejects_non_mono_or_non_16bit(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = ModulateSTTProvider(api_key=fake_api_key)
    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=2,
        sample_width=2,
        sample_rate=16000,
    )

    assert result.error is not None
    assert "mono 16-bit" in result.error
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_modulate_server_error_surfaces(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [{"type": "error", "error": "unsupported audio format"}]
    provider = ModulateSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.modulate.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error == "unsupported audio format"
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_modulate_empty_stream_is_error(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A stream that closes without ``done`` is an error, not an empty success."""
    provider = ModulateSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.modulate.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "done" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_modulate_truncated_stream_is_error(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A close after a final but before ``done`` is flagged, not reported clean."""
    events: list[Any] = [
        {"type": "partial_utterance", "partial_utterance": {"text": "hello"}},
        {"type": "utterance", "utterance": {"text": "hello world"}},
    ]
    provider = ModulateSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.modulate.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "done" in result.error


@pytest.mark.asyncio
async def test_modulate_connection_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """A connect failure surfaces as result.error, not a silent empty success."""
    provider = ModulateSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.modulate.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_modulate_surfaces_send_failure(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A send failure inside the streaming task is surfaced, not swallowed by gather."""

    def _raise_on_audio(msg: object) -> None:
        if isinstance(msg, (bytes, bytearray)):
            raise RuntimeError("send boom")

    ws = FakeWebSocket([], on_send=_raise_on_audio)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    provider = ModulateSTTProvider(api_key=fake_api_key)
    with patch("coval_bench.providers.stt.modulate.ws_client.connect", return_value=cm):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "send boom" in result.error
    assert result.complete_transcript is None
