# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.mistral (MistralSTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.mistral import MistralSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def _fake_connect(
    events: list[Any],
    *,
    captured: dict[str, Any] | None = None,
) -> Any:
    ws = FakeWebSocket(events)
    if captured is not None:
        captured["ws"] = ws
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_mistral_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """Deltas drive TTFT; transcription.done carries the final transcript."""
    events = load_fixture_events("mistral", "events-success")
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
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
    assert result.first_token_content == "hello"  # noqa: S105 - transcript fixture text
    assert result.complete_transcript == "hello world how are you"
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_mistral_missing_done_falls_back_to_deltas(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """Stream closing without transcription.done keeps the joined deltas but no final time."""
    events: list[Any] = [
        {
            "type": "session.created",
            "session": {"request_id": "r1", "audio_format": None},
        },
        {"type": "transcription.text.delta", "text": "hello"},
        {"type": "transcription.text.delta", "text": " world"},
    ]
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.complete_transcript == "hello world"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_mistral_empty_deltas_ignored_for_ttft(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """Whitespace-only deltas never stamp TTFT or first_token_content."""
    events: list[Any] = [
        {
            "type": "session.created",
            "session": {"request_id": "r2", "audio_format": None},
        },
        {"type": "transcription.text.delta", "text": " "},
        {"type": "transcription.text.delta", "text": "hi"},
        {
            "type": "transcription.done",
            "model": "voxtral-mini-transcribe-realtime-2602",
            "text": "hi",
            "language": "en",
            "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
        },
    ]
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
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
    assert result.first_token_content == "hi"  # noqa: S105 - transcript fixture text
    assert result.complete_transcript == "hi"


@pytest.mark.asyncio
async def test_mistral_partial_transcripts_accumulate(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """Each non-empty delta is recorded in partial_transcripts in arrival order."""
    events = load_fixture_events("mistral", "events-success")
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
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
    assert result.partial_transcripts == ["hello", " world", " how are you"]


@pytest.mark.asyncio
async def test_mistral_first_done_wins(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """A rogue early transcription.done ends the session; later events are ignored.

    transcription.done is session-terminal in the protocol (the vendor SDK also
    stops on the first one), so a mid-utterance done must not be double-counted
    or merged with anything the server sends afterwards.
    """
    events: list[Any] = [
        {
            "type": "session.created",
            "session": {"request_id": "r4", "audio_format": None},
        },
        {"type": "transcription.text.delta", "text": "hello world"},
        {
            "type": "transcription.done",
            "model": "voxtral-mini-transcribe-realtime-2602",
            "text": "hello world",
            "language": "en",
            "usage": {"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2},
        },
        {"type": "transcription.text.delta", "text": " how are you"},
        {
            "type": "transcription.done",
            "model": "voxtral-mini-transcribe-realtime-2602",
            "text": "hello world how are you",
            "language": "en",
            "usage": {"prompt_tokens": 0, "completion_tokens": 5, "total_tokens": 5},
        },
    ]
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
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
    assert result.partial_transcripts == ["hello world"]
    assert result.audio_to_final_seconds is not None


@pytest.mark.asyncio
async def test_mistral_websocket_connect_setup(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events = load_fixture_events("mistral", "events-success")
    provider = MistralSTTProvider(api_key=fake_api_key)
    captured: dict[str, Any] = {}

    def connect_side_effect(
        url: str,
        additional_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        captured["url"] = url
        captured["additional_headers"] = dict(additional_headers or {})
        return _fake_connect(events, captured=captured)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    parsed = urlparse(captured["url"])
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.mistral.ai"
    assert parsed.path == "/v1/audio/transcriptions/realtime"
    query = parse_qs(parsed.query)
    assert query.get("model") == ["voxtral-mini-transcribe-realtime-2602"]
    auth = captured["additional_headers"].get("Authorization")
    assert auth == f"Bearer {fake_api_key.get_secret_value()}"

    sent_json = [json.loads(msg) for msg in captured["ws"]._sent if isinstance(msg, str)]
    assert sent_json[0]["type"] == "session.update"
    assert sent_json[0]["session"]["audio_format"] == {
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
    }
    appends = [msg for msg in sent_json if msg["type"] == "input_audio.append"]
    assert appends, "expected input_audio.append frames to be sent"
    sent_audio = b"".join(base64.b64decode(msg["audio"]) for msg in appends)
    assert sent_audio == audio_pcm_bytes
    assert sent_json[-2]["type"] == "input_audio.flush"
    assert sent_json[-1]["type"] == "input_audio.end"


@pytest.mark.asyncio
async def test_mistral_empty_session_errors(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """No deltas and an empty transcription.done is an error, not a final.

    Models a degraded/dead session (handshake ok, ASR backend emits nothing). It
    must fail — not stamp a spurious audio_to_final from the empty terminal event.
    """
    events: list[Any] = [
        {
            "type": "session.created",
            "session": {"request_id": "r3", "audio_format": None},
        },
        {
            "type": "transcription.done",
            "model": "voxtral-mini-transcribe-realtime-2602",
            "text": "",
            "language": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    ]
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
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
    assert "empty transcript" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_mistral_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("mistral", "events-error")
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
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
    assert "invalid api key" in result.error
    assert result.complete_transcript is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_mistral_handshake_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """An error event before session.created fails the run."""
    events: list[Any] = [
        {"type": "error", "error": {"code": "unauthorized", "message": "invalid api key"}},
    ]
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
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
    assert "invalid api key" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_handshake_closes_before_created(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """WS exhausting before session.created surfaces an error."""
    events: list[Any] = [b"\x00\x00"]
    provider = MistralSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.mistral.ws_client.connect",
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
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_session_created_deadline(fake_api_key: SecretStr) -> None:
    """_await_session_created raises once the deadline elapses without session.created."""
    provider = MistralSTTProvider(api_key=fake_api_key)
    ws = FakeWebSocket([b"\x00\x00"])

    times = [0.0, 0.0, 11.0]

    def fake_monotonic() -> float:
        return times.pop(0) if times else 11.0

    with (
        patch(
            "coval_bench.providers.stt.mistral.time.monotonic",
            side_effect=fake_monotonic,
        ),
        pytest.raises(TimeoutError, match="session.created"),
    ):
        await provider._await_session_created(ws)


def test_mistral_provider_name() -> None:
    provider = MistralSTTProvider(api_key=SecretStr("test"))
    assert provider.name == "mistral-voxtral-mini-transcribe-realtime-2602"


def test_mistral_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Mistral model"):
        MistralSTTProvider(api_key=SecretStr("k"), model="bad-model")


@pytest.mark.asyncio
async def test_mistral_rejects_stereo(fake_api_key: SecretStr) -> None:
    provider = MistralSTTProvider(api_key=fake_api_key)

    result = await provider.measure_ttft(
        audio_data=b"\x00" * 16,
        channels=2,
        sample_width=2,
        sample_rate=16000,
    )

    assert result.error is not None
    assert "mono" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_rejects_non_pcm16(fake_api_key: SecretStr) -> None:
    provider = MistralSTTProvider(api_key=fake_api_key)

    result = await provider.measure_ttft(
        audio_data=b"\x00" * 16,
        channels=1,
        sample_width=1,
        sample_rate=16000,
    )

    assert result.error is not None
    assert "16-bit PCM" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_rejects_unsupported_sample_rate(fake_api_key: SecretStr) -> None:
    provider = MistralSTTProvider(api_key=fake_api_key)

    result = await provider.measure_ttft(
        audio_data=b"\x00" * 16,
        channels=1,
        sample_width=2,
        sample_rate=12345,
    )

    assert result.error is not None
    assert "12345" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_rejects_non_positive_realtime_resolution(
    fake_api_key: SecretStr,
) -> None:
    provider = MistralSTTProvider(api_key=fake_api_key)

    result = await provider.measure_ttft(
        audio_data=b"\x00" * 16,
        channels=1,
        sample_width=2,
        sample_rate=16000,
        realtime_resolution=0,
    )

    assert result.error is not None
    assert "realtime_resolution must be > 0" in result.error
    assert result.ttft_seconds is None
