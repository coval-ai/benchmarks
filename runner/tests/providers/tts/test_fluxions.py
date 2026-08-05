# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Fluxions VUI WebSocket TTS provider."""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts import fluxions as fluxions_module
from coval_bench.providers.tts.fluxions import FluxionsTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_VOICE = "maeve"
_VOICE_ID = "maeve.h8ff7e07da"
_CATALOG = {"maeve": _VOICE_ID, "abraham": "abraham.h8ff7e07da"}


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        fluxions_api_key=SecretStr("test-api-key"),
    )


def _events(pcm_chunks: list[bytes]) -> list[str | bytes]:
    events: list[str | bytes] = [json.dumps({"type": "start"})]
    events.extend(pcm_chunks)
    events.append(json.dumps({"type": "done"}))
    return events


@pytest.fixture()
def fluxions_settings() -> Settings:
    return _settings()


@pytest.fixture(autouse=True)
def _seeded_catalog() -> Iterator[None]:
    """Pre-seed the voice catalog so no test reaches the network to resolve a voice."""
    original = dict(fluxions_module._VOICE_IDS)
    fluxions_module._VOICE_IDS.clear()
    fluxions_module._VOICE_IDS.update(_CATALOG)
    yield
    fluxions_module._VOICE_IDS.clear()
    fluxions_module._VOICE_IDS.update(original)


@pytest.mark.asyncio
async def test_fluxions_tts_happy_path(fluxions_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice=_VOICE)

    with patch("coval_bench.providers.tts.fluxions.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Fluxions")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "fluxions"
    assert result.model == "vui"
    # The row records the stable registry voice, not the hashed catalog id.
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fluxions_tts_url_and_speak_frame(fluxions_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["headers"] = kwargs.get("additional_headers")
        return ws

    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.fluxions.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert captured["url"] == "wss://api.fluxions.ai/vui/v1/tts/ws"
    assert captured["headers"] == {"Authorization": "Bearer test-api-key"}
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent == [
        {
            "type": "speak",
            "voice": _VOICE_ID,
            "input": "Hello world",
            "verify_chunks": False,
        }
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fluxions_tts_resolves_voice_from_catalog(fluxions_settings: Settings) -> None:
    """An unseen bare name triggers one catalog fetch, and the id it maps to is used."""
    fluxions_module._VOICE_IDS.clear()
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice="abraham")

    with (
        patch(
            "coval_bench.providers.tts.fluxions._load_voice_catalog",
            return_value=_CATALOG,
        ) as load,
        patch("coval_bench.providers.tts.fluxions.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    load.assert_awaited_once()
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent[0]["voice"] == "abraham.h8ff7e07da"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fluxions_tts_hashed_voice_passes_through(fluxions_settings: Settings) -> None:
    """A fully-qualified voice id skips resolution entirely."""
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice="harry.hdeadbeef")

    with (
        patch("coval_bench.providers.tts.fluxions._load_voice_catalog") as load,
        patch("coval_bench.providers.tts.fluxions.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    load.assert_not_called()
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent[0]["voice"] == "harry.hdeadbeef"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fluxions_tts_unknown_voice_is_sent_verbatim(fluxions_settings: Settings) -> None:
    """An unresolvable name goes to the server as-is so the error comes from Fluxions."""
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice="nosuchvoice")

    with (
        patch(
            "coval_bench.providers.tts.fluxions._load_voice_catalog",
            return_value=_CATALOG,
        ),
        patch("coval_bench.providers.tts.fluxions.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent[0]["voice"] == "nosuchvoice"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fluxions_tts_error_event(fluxions_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"type": "start"}),
        json.dumps({"type": "error", "message": "voice not found"}),
    ]
    ws = FakeWebSocket(events)
    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice=_VOICE)

    with patch("coval_bench.providers.tts.fluxions.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "voice not found" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_fluxions_tts_skips_empty_frames(fluxions_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"type": "start"}),
        b"",
        make_pcm_bytes(240),
        json.dumps({"type": "done"}),
    ]
    ws = FakeWebSocket(events)
    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice=_VOICE)

    with patch("coval_bench.providers.tts.fluxions.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fluxions_tts_ttfa_on_first_chunk(fluxions_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_events(chunks))
    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.fluxions.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.fluxions.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_fluxions_tts_invalid_model_raises(fluxions_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Fluxions TTS model"):
        FluxionsTTSProvider(fluxions_settings, model="not-a-model", voice=_VOICE)


def test_fluxions_tts_missing_voice_raises(fluxions_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires a voice"):
        FluxionsTTSProvider(fluxions_settings, model="vui", voice="")


def test_fluxions_tts_missing_api_key_raises(fluxions_settings: Settings) -> None:
    fluxions_settings.fluxions_api_key = None
    with pytest.raises(ValueError, match="fluxions_api_key"):
        FluxionsTTSProvider(fluxions_settings, model="vui", voice=_VOICE)


def test_fluxions_tts_provider_name(fluxions_settings: Settings) -> None:
    provider = FluxionsTTSProvider(fluxions_settings, model="vui", voice=_VOICE)
    assert provider.name == "fluxions-vui"
    assert provider.model == "vui"


@pytest.mark.asyncio
async def test_fluxions_warmup_caches_catalog(fluxions_settings: Settings) -> None:
    fluxions_module._VOICE_IDS.clear()

    with patch(
        "coval_bench.providers.tts.fluxions._load_voice_catalog",
        return_value=_CATALOG,
    ):
        await FluxionsTTSProvider.warmup(fluxions_settings)

    assert fluxions_module._VOICE_IDS == _CATALOG
