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
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.revai import RevAISTTProvider
from tests.providers.fakes import FakeWebSocket, fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings


def make_provider() -> RevAISTTProvider:
    return RevAISTTProvider(make_settings())


# ---------------------------------------------------------------------------
# Happy path — final text elements concatenated, punct dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revai_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("revai", "events-success")
    provider = RevAISTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
    ) as connect:
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

    url = connect.call_args.args[0]
    assert url.startswith("wss://api.rev.ai/speechtotext/v1/stream?")
    assert "access_token=test-api-key-00000000000000000000" in url
    assert "transcriber=machine_v2" in url


@pytest.mark.asyncio
async def test_revai_ttft_fires_on_first_partial(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """TTFT is time-to-first-word: it must fire on a partial, before any final."""
    events: list[Any] = [
        {"type": "connected"},
        {"type": "partial", "elements": [{"type": "text", "value": "hello"}]},
        {"type": "final", "elements": [{"type": "text", "value": "hello world"}]},
    ]
    provider = RevAISTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """Only ``final`` messages form the transcript; partials are dropped."""
    events: list[Any] = [
        {"type": "connected"},
        {"type": "partial", "elements": [{"type": "text", "value": "wrong guess"}]},
        {"type": "final", "elements": [{"type": "text", "value": "hello"}]},
        {"type": "partial", "elements": [{"type": "text", "value": "worng"}]},
        {"type": "final", "elements": [{"type": "text", "value": "world"}]},
    ]
    provider = RevAISTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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
async def test_revai_tail_partial_uses_latest_not_longest(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """A downward-revised trailing partial (no final) recovers the latest guess.

    Rev's partials are cumulative per segment, so a walked-back hypothesis
    ("hello world foo" -> "hello world") must win over the stale longer one.
    """
    events: list[Any] = [
        {"type": "connected"},
        {"type": "partial", "elements": [{"type": "text", "value": "hello world foo"}]},
        {"type": "partial", "elements": [{"type": "text", "value": "hello world"}]},
    ]
    provider = RevAISTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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


@pytest.mark.asyncio
async def test_revai_sends_eos_sentinel(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """Trailing silence, then an ``EOS`` text frame, follow the PCM chunks."""
    sent: list[Any] = []
    provider = RevAISTTProvider(stt_settings)
    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect([{"type": "connected"}], sent),
    ):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert sent[-1] == "EOS"
    audio = [m for m in sent if isinstance(m, bytes)]
    assert b"".join(audio).startswith(audio_pcm_bytes)
    assert len(b"".join(audio)) > len(audio_pcm_bytes)


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
        RevAISTTProvider(make_settings(), model="machine_v3")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="revai_api_key is required"):
        RevAISTTProvider(make_settings(revai_api_key=None))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revai_rejects_non_mono_or_non_16bit(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    provider = RevAISTTProvider(stt_settings)
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
async def test_revai_empty_stream(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = RevAISTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect([]),
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
async def test_revai_connection_error(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """A connect failure surfaces as result.error, not a silent empty success."""
    provider = RevAISTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
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
async def test_revai_surfaces_send_failure(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """A send failure inside the streaming task is surfaced, not swallowed by gather."""

    def _raise_on_audio(msg: object) -> None:
        if isinstance(msg, (bytes, bytearray)):
            raise RuntimeError("send boom")

    cm = fake_connect([])
    cm.__aenter__.return_value = FakeWebSocket([], on_send=_raise_on_audio)

    provider = RevAISTTProvider(stt_settings)
    with patch("coval_bench.providers.stt._streaming.ws_client.connect", return_value=cm):
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
