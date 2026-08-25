# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs TTS provider — WebSocket streaming.

eleven_v3_conversational speaks the Text to Dialogue WebSocket; the other
models speak the per-voice stream-input WebSocket. TTFA is measured from the
text submit to the first PCM frame; session setup (connect plus the setup
frame) stays out of the measurement.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000

_DIALOGUE_WS_URL = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
_STREAM_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
_OUTPUT_FORMAT = "pcm_24000"
_LAST_FRAMES_KEPT = 3


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider over per-model WebSocket endpoints."""

    _VALID_MODELS = frozenset({"eleven_v3_conversational", "eleven_flash_v2_5"})
    _DIALOGUE_MODELS = frozenset({"eleven_v3_conversational"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Unsupported ElevenLabs model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._model = model
        self._voice = voice

        api_key_secret = settings.elevenlabs_api_key
        if api_key_secret is None:
            raise ValueError("elevenlabs_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"elevenlabs-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _session_frames(self, text: str) -> tuple[str, list[str], list[str]]:
        """URL, pre-clock setup frames, and timed text frames for this model."""
        if self._model in self._DIALOGUE_MODELS:
            url = f"{_DIALOGUE_WS_URL}?model_id={self._model}&output_format={_OUTPUT_FORMAT}"
            setup = [json.dumps({"voices": [self._voice]})]
            inputs = [
                json.dumps(
                    {"inputs": [{"text": text, "voice_id": self._voice}], "close_socket": True}
                )
            ]
        else:
            url = (
                f"{_STREAM_WS_URL.format(voice_id=self._voice)}"
                f"?model_id={self._model}&output_format={_OUTPUT_FORMAT}"
            )
            # The empty text frame ends the input stream and flushes synthesis.
            setup = [json.dumps({"text": " "})]
            inputs = [json.dumps({"text": text + " "}), json.dumps({"text": ""})]
        return url, setup, inputs

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        last_frames: list[str] = []
        start: float | None = None
        first_chunk_at: float | None = None

        url, setup_frames, text_frames = self._session_frames(text)

        try:
            async with ws_client.connect(
                url, additional_headers={"xi-api-key": self._api_key}
            ) as ws:
                for frame in setup_frames:
                    await ws.send(frame)

                start = time.monotonic()
                for frame in text_frames:
                    await ws.send(frame)

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        event: dict[str, Any] = json.loads(raw)
                    except json.JSONDecodeError:
                        last_frames.append(raw)
                        del last_frames[:-_LAST_FRAMES_KEPT]
                        continue

                    if event.get("error"):
                        raise RuntimeError(f"{event.get('error')}: {event.get('message')}")

                    audio_b64 = event.get("audio")
                    if audio_b64:
                        chunk = base64.b64decode(audio_b64)
                        if chunk:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(chunk)
                    else:
                        last_frames.append(raw)
                        del last_frames[:-_LAST_FRAMES_KEPT]

                    # Dialogue closes with snake_case is_final; stream-input
                    # with camelCase isFinal.
                    if event.get("is_final") or event.get("isFinal"):
                        break

        except Exception as exc:
            logger.warning(
                "elevenlabs_error", provider="elevenlabs", model=self._model, exc_info=exc
            )
            return finalize_tts_result(
                provider="elevenlabs",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                last_frames=last_frames,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="elevenlabs",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            last_frames=last_frames,
        )
