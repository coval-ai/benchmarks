# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.tts.google (GoogleTTSProvider).

The google-cloud-texttospeech package is optional (extra: ``google-tts``).
Tests are skipped when it is not installed.

To run these tests locally:
    uv sync --extra google-tts
    uv run pytest tests/providers/tts/test_google.py -v
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

texttospeech = pytest.importorskip(
    "google.cloud.texttospeech",
    reason="google-cloud-texttospeech not installed; install with: uv sync --extra google-tts",
)

from coval_bench.config import Settings  # noqa: E402
from coval_bench.providers.tts import google as google_tts  # noqa: E402
from coval_bench.providers.tts.google import SAMPLE_RATE, GoogleTTSProvider  # noqa: E402

from .conftest import make_pcm_bytes  # noqa: E402

CHIRP_MODEL = "chirp-3-hd"
CHIRP_VOICE = "en-US-Chirp3-HD-Kore"
GEMINI_MODEL = "gemini-2.5-flash-tts"
GEMINI_VOICE = "Kore"


@pytest.fixture(autouse=True)
def _reset_shared_client() -> Generator[None, None, None]:
    """Isolate the module-level shared client between tests."""
    google_tts._shared_client = None
    yield
    google_tts._shared_client = None


def _make_client(responses: list[Any] | None = None) -> tuple[MagicMock, dict[str, Any]]:
    """Mock client whose streaming_synthesize consumes and captures requests."""
    captured: dict[str, Any] = {}
    client = MagicMock()

    def _fake_streaming_synthesize(requests: Any) -> Any:
        captured["requests"] = list(requests)
        return iter(responses or [])

    client.streaming_synthesize.side_effect = _fake_streaming_synthesize
    return client, captured


def _audio_responses(n: int = 3) -> list[Any]:
    return [
        texttospeech.StreamingSynthesizeResponse(audio_content=make_pcm_bytes(480))
        for _ in range(n)
    ]


@pytest.mark.asyncio
async def test_google_chirp_success(fake_settings: Settings, sample_text: str) -> None:
    client, captured = _make_client(_audio_responses())

    with patch.object(google_tts, "_get_shared_client", return_value=client):
        provider = GoogleTTSProvider(fake_settings, model=CHIRP_MODEL, voice=CHIRP_VOICE)
        result = await provider.synthesize(sample_text)

    assert result.error is None
    assert result.provider == "google"
    assert result.model == CHIRP_MODEL
    assert result.voice == CHIRP_VOICE
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None and result.audio_path.exists()
    result.audio_path.unlink()

    requests = captured["requests"]
    assert len(requests) == 2
    config = requests[0].streaming_config
    assert config.voice.name == CHIRP_VOICE
    assert config.voice.language_code == "en-US"
    assert config.voice.model_name == ""
    assert config.streaming_audio_config.audio_encoding == texttospeech.AudioEncoding.PCM
    assert config.streaming_audio_config.sample_rate_hertz == SAMPLE_RATE
    assert requests[1].input.text == sample_text


@pytest.mark.asyncio
async def test_google_gemini_voice_params(fake_settings: Settings, sample_text: str) -> None:
    client, captured = _make_client(_audio_responses())

    with patch.object(google_tts, "_get_shared_client", return_value=client):
        provider = GoogleTTSProvider(fake_settings, model=GEMINI_MODEL, voice=GEMINI_VOICE)
        result = await provider.synthesize(sample_text)

    assert result.error is None
    voice = captured["requests"][0].streaming_config.voice
    assert voice.name == GEMINI_VOICE
    assert voice.language_code == "en-US"
    assert voice.model_name == GEMINI_MODEL
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_google_stream_error(fake_settings: Settings, sample_text: str) -> None:
    client = MagicMock()
    client.streaming_synthesize.side_effect = RuntimeError("stream exploded")

    with patch.object(google_tts, "_get_shared_client", return_value=client):
        provider = GoogleTTSProvider(fake_settings, model=CHIRP_MODEL, voice=CHIRP_VOICE)
        result = await provider.synthesize(sample_text)

    assert result.error is not None and "stream exploded" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_google_no_audio(fake_settings: Settings, sample_text: str) -> None:
    client, _ = _make_client([])

    with patch.object(google_tts, "_get_shared_client", return_value=client):
        provider = GoogleTTSProvider(fake_settings, model=CHIRP_MODEL, voice=CHIRP_VOICE)
        result = await provider.synthesize(sample_text)

    assert result.error is None
    assert result.ttfa_ms is None
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_google_unsupported_model(fake_settings: Settings, sample_text: str) -> None:
    client, _ = _make_client()

    with patch.object(google_tts, "_get_shared_client", return_value=client):
        provider = GoogleTTSProvider(fake_settings, model="chirp-2-hd", voice=CHIRP_VOICE)
        result = await provider.synthesize(sample_text)

    assert result.error is not None and "Unsupported" in result.error
    assert result.audio_path is None
    client.streaming_synthesize.assert_not_called()


def test_google_properties(fake_settings: Settings) -> None:
    provider = GoogleTTSProvider(fake_settings, model=CHIRP_MODEL, voice=CHIRP_VOICE)
    assert provider.name == "google-chirp-3-hd"
    assert provider.model == CHIRP_MODEL


@pytest.mark.asyncio
async def test_google_warmup_opens_channel(fake_settings: Settings) -> None:
    client = MagicMock()

    with patch.object(google_tts, "_get_shared_client", return_value=client):
        await GoogleTTSProvider.warmup(settings=fake_settings)

    client.list_voices.assert_called_once_with(language_code="en-US")
