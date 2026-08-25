# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Atlas WebSocket streaming TTS provider.

The interesting cases are not the happy path but the ways an Atlas row could be
flattered: a TTFA clock that starts somewhere other than the text submit, leading
silence hiding inside a bare arrival time, a partial clip scored against a full
prompt, and a vendor-side sample-rate change rescaling every published duration.
"""

from __future__ import annotations

import json
import struct
import wave
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.atlas import SAMPLE_RATE, VALID_VOICES, AtlasTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_MODEL = "atlas-tts"
_VOICE = "dax"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider(fake_settings: Settings, voice: str = _VOICE) -> AtlasTTSProvider:
    return AtlasTTSProvider(fake_settings, model=_MODEL, voice=voice)


def _frame(frame_type: str, **fields: object) -> str:
    return json.dumps({"type": frame_type, **fields})


def _session(
    audio: list[bytes],
    *,
    ready_rate: int | None = SAMPLE_RATE,
    audio_done: dict[str, object] | None = None,
    closed_early: bool = False,
) -> list[str | bytes]:
    """A well-formed Atlas session: ready -> audio.start -> pcm -> done."""
    ready: dict[str, object] = {}
    if ready_rate is not None:
        ready["sample_rate"] = ready_rate
    messages: list[str | bytes] = [
        _frame("ready", **ready),
        _frame("audio.start", sample_rate=ready_rate) if ready_rate else _frame("audio.start"),
        *audio,
        _frame("audio.done", **(audio_done or {})),
    ]
    if not closed_early:
        messages.append(_frame("session.done"))
    return messages


def _silence(frames: int) -> bytes:
    return struct.pack("<h", 0) * frames


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_atlas_happy_path(fake_settings: Settings) -> None:
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([make_pcm_bytes()]))

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.provider == "atlas"
    assert result.model == _MODEL
    assert result.voice == _VOICE
    assert result.ttfa_ms is not None
    assert result.audio_path is not None


@pytest.mark.asyncio
async def test_requests_pcm_and_the_configured_voice(fake_settings: Settings) -> None:
    """Containers only complete at end of synthesis, so PCM is required for TTFA."""
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([make_pcm_bytes()]))

    with patch("websockets.asyncio.client.connect", return_value=ws):
        await provider.synthesize("Hello world")

    start = json.loads(ws.sent[0])
    assert start["type"] == "start"
    assert start["response_format"] == "pcm"
    assert start["voice"] == _VOICE


@pytest.mark.asyncio
async def test_text_is_submitted_only_after_ready(fake_settings: Settings) -> None:
    """The frame order the TTFA boundary depends on: start, then text on ready."""
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([make_pcm_bytes()]))

    with patch("websockets.asyncio.client.connect", return_value=ws):
        await provider.synthesize("Hello world")

    assert [json.loads(frame)["type"] for frame in ws.sent] == ["start", "text", "done"]


# ---------------------------------------------------------------------------
# TTFA boundary — parity with the other WebSocket TTS providers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttfa_excludes_session_setup(fake_settings: Settings) -> None:
    """t0 is the text submit, so setup before ``ready`` cannot inflate TTFA.

    Every WebSocket TTS provider here starts its clock immediately before the text
    submit - minimax waits for ``task_started``, elevenlabs sends its setup frame
    first - so connect, the ``start`` frame and the ``ready`` ack all stay out.
    Five seconds burned during setup must therefore not reach TTFA.
    """
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([make_pcm_bytes()]))
    clock = {"t": 1000.0}
    original_send = ws.send

    async def send(data: str | bytes) -> None:
        if isinstance(data, str) and json.loads(data).get("type") == "start":
            clock["t"] += 5.0
        await original_send(data)

    ws.send = send  # type: ignore[method-assign]

    with (
        patch("websockets.asyncio.client.connect", return_value=ws),
        patch("coval_bench.providers.tts.atlas.time.monotonic", lambda: clock["t"]),
    ):
        result = await provider.synthesize("Hello world")

    assert result.ttfa_ms is not None
    assert result.ttfa_ms < 5000.0


@pytest.mark.asyncio
async def test_leading_silence_is_split_out_not_swallowed(fake_settings: Settings) -> None:
    """An instant chunk of silence is the cheapest way to win a TTFA benchmark."""
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([_silence(SAMPLE_RATE // 10) + make_pcm_bytes()]))

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    # The detector uses a 10 ms RMS window, so the reported onset is the start of
    # the frame straddling the boundary - up to one frame early, never late.
    assert result.leading_silence_ms is not None
    assert 90.0 <= result.leading_silence_ms <= 100.0
    assert result.ttfa_ms is not None
    assert result.ttfa_ms >= result.leading_silence_ms


@pytest.mark.asyncio
async def test_all_silence_is_a_failure_not_a_fast_ttfa(fake_settings: Settings) -> None:
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([_silence(SAMPLE_RATE // 2)]))

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    assert result.error is not None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Sample rate — pinned per provider, mismatch surfaced
# ---------------------------------------------------------------------------


def test_sample_rate_is_pinned() -> None:
    """Pinned per provider as everywhere else in this package; 24 kHz measured."""
    assert SAMPLE_RATE == 24000


@pytest.mark.asyncio
async def test_declared_rate_mismatch_does_not_rescale(fake_settings: Settings) -> None:
    """A vendor-side rate change must surface, not silently rescale durations."""
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([make_pcm_bytes()], ready_rate=48000))

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.audio_path is not None
    # The WAV is written at the pinned rate regardless of what Atlas declared.
    with wave.open(str(result.audio_path)) as handle:
        assert handle.getframerate() == SAMPLE_RATE


@pytest.mark.asyncio
async def test_missing_declared_rate_is_tolerated(fake_settings: Settings) -> None:
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([make_pcm_bytes()], ready_rate=None))

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.ttfa_ms is not None


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_sentence_error_flag_fails_the_result(fake_settings: Settings) -> None:
    """A truncated clip must not be scored against the full prompt."""
    provider = _provider(fake_settings)
    ws = FakeWebSocket(
        _session([make_pcm_bytes()], audio_done={"error": True, "sentence_index": 2})
    )

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    assert result.error is not None
    assert "sentence 2" in result.error


@pytest.mark.asyncio
async def test_error_frame_fails_the_result(fake_settings: Settings) -> None:
    provider = _provider(fake_settings)
    ws = FakeWebSocket([_frame("ready", sample_rate=SAMPLE_RATE), _frame("error", message="boom")])

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    assert result.error is not None
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_stream_closed_before_session_done_fails(fake_settings: Settings) -> None:
    """Silent truncation would otherwise score a partial clip as a success."""
    provider = _provider(fake_settings)
    ws = FakeWebSocket(_session([make_pcm_bytes()], closed_early=True))

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    assert result.error is not None
    assert "session.done" in result.error


@pytest.mark.asyncio
async def test_unknown_frames_are_ignored(fake_settings: Settings) -> None:
    provider = _provider(fake_settings)
    messages = _session([make_pcm_bytes()])
    messages.insert(1, _frame("some.future.frame", detail="ignored"))
    ws = FakeWebSocket(messages)

    with patch("websockets.asyncio.client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.ttfa_ms is not None


# ---------------------------------------------------------------------------
# Construction and registration
# ---------------------------------------------------------------------------


def test_dax_is_a_listed_voice() -> None:
    """``dax`` is a stock voice from /v1/models, not a per-account clone."""
    assert "dax" in VALID_VOICES


def test_valid_voices_match_the_published_list() -> None:
    assert VALID_VOICES[:3] == ["capella", "dax", "enzo"]
    assert len(VALID_VOICES) == 14


@pytest.mark.asyncio
async def test_unknown_voice_falls_back_to_dax(fake_settings: Settings) -> None:
    assert _provider(fake_settings, voice="not-a-voice")._voice == "dax"


def test_invalid_model_is_rejected(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Atlas TTS model"):
        AtlasTTSProvider(fake_settings, model="not-a-model", voice=_VOICE)


def test_missing_key_is_rejected() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        atlas_api_key=None,
    )
    with pytest.raises(ValueError, match="atlas_api_key"):
        AtlasTTSProvider(settings, model=_MODEL, voice=_VOICE)


def test_registry_entry_is_early_access_off_arena_and_shared_inference() -> None:
    """Atlas proxies an undisclosed upstream, so it is not published as first-party.

    Pinned as a test because these three fields are the only thing keeping a
    possibly non-independent row off the public board and out of the blind
    voice-quality A/B.
    """
    from coval_bench.registries import MODEL_REGISTRY
    from coval_bench.registries.models import Source

    rows = [m for m in MODEL_REGISTRY if m.provider == "atlas"]
    assert len(rows) == 1
    assert rows[0].model == _MODEL
    assert rows[0].voice == _VOICE
    assert rows[0].collected is True
    assert rows[0].published is False
    assert rows[0].source is Source.SHARED_INFERENCE
    assert rows[0].arena_enabled is False


def test_atlas_has_a_provider_env_entry() -> None:
    """Without this, publishing Atlas breaks the arena key parity check."""
    from coval_bench.registries.provider_keys import PROVIDER_ENV

    assert PROVIDER_ENV["atlas"] == "ATLAS_API_KEY"
