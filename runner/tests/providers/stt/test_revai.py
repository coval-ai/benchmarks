# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.revai (RevAISTTProvider).

All tests use FakeWebSocket — no live network calls. The event fixtures mirror
the Rev AI streaming wire protocol: a ``connected`` handshake, then ``partial``
and ``final`` messages carrying an ``elements`` array of ``text``/``punct``
items. The transcript is the concatenation of the ``text`` elements from every
``final`` message; ``punct`` elements are dropped for WER parity.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.revai import RevAISTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> RevAISTTProvider:
    return RevAISTTProvider(api_key=SecretStr("test-key-revai"))


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path — final text elements concatenated, punct dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revai_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("revai", "events-success")
    provider = RevAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.revai.ws_client.connect",
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
async def test_revai_ttft_fires_on_first_partial(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """TTFT is time-to-first-word: it must fire on a partial, before any final."""
    events: list[Any] = [
        {"type": "connected"},
        {"type": "partial", "elements": [{"type": "text", "value": "hello"}]},
        {"type": "final", "elements": [{"type": "text", "value": "hello world"}]},
    ]
    provider = RevAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.revai.ws_client.connect",
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
async def test_revai_excludes_partials_from_transcript(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Only ``final`` messages form the transcript; partials are dropped."""
    events: list[Any] = [
        {"type": "connected"},
        {"type": "partial", "elements": [{"type": "text", "value": "wrong guess"}]},
        {"type": "final", "elements": [{"type": "text", "value": "hello"}]},
        {"type": "partial", "elements": [{"type": "text", "value": "worng"}]},
        {"type": "final", "elements": [{"type": "text", "value": "world"}]},
    ]
    provider = RevAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.revai.ws_client.connect",
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
async def test_revai_sends_eos_sentinel(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """End-of-audio is signalled with an ``EOS`` text frame after the PCM chunks."""
    ws = FakeWebSocket([{"type": "connected"}])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    provider = RevAISTTProvider(api_key=fake_api_key)
    with patch("coval_bench.providers.stt.revai.ws_client.connect", return_value=cm):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert ws._sent[-1] == "EOS"


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "revai"


def test_provider_model() -> None:
    assert make_provider().model == "reverb"


# ---------------------------------------------------------------------------
# Invalid construction
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Rev AI model"):
        RevAISTTProvider(api_key=SecretStr("k"), model="machine_v3")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="revai_api_key is required"):
        RevAISTTProvider(api_key=None)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revai_rejects_non_mono_or_non_16bit(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = RevAISTTProvider(api_key=fake_api_key)
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
async def test_revai_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = RevAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.revai.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_revai_connection_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """A connect failure surfaces as result.error, not a silent empty success."""
    provider = RevAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.revai.ws_client.connect",
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
async def test_revai_surfaces_send_failure(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """A send failure inside the streaming task is surfaced, not swallowed by gather."""

    def _raise_on_audio(msg: object) -> None:
        if isinstance(msg, (bytes, bytearray)):
            raise RuntimeError("send boom")

    ws = FakeWebSocket([], on_send=_raise_on_audio)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    provider = RevAISTTProvider(api_key=fake_api_key)
    with patch("coval_bench.providers.stt.revai.ws_client.connect", return_value=cm):
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
