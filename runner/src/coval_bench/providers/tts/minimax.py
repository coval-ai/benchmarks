# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""MiniMax TTS provider (Speech 2.8 line over WebSocket).

Wire protocol on wss://api.minimax.io/ws/v1/t2a_v2, Bearer auth:
  connect → recv connected_success → send task_start (model + voice/audio)
  → recv task_started → send task_continue(text) + task_finish
  → recv task_continued frames (hex-encoded PCM) until is_final → close

The TTFA clock starts at task_continue, after the task_started ack, so the
session-setup round trip is excluded (parity with providers that configure the
model on the WS handshake itself).
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("speech-2.8-hd", "speech-2.8-turbo")
_WS_URL = "wss://api.minimax.io/ws/v1/t2a_v2"
_SAMPLE_RATE = 44100
_MAX_WS_SIZE = 16 * 1024 * 1024


class MinimaxTTSProvider(TTSProvider):
    """MiniMax TTS provider using WebSocket streaming."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid MiniMax TTS model {model!r}. Valid: {_VALID_MODELS}")
        if not voice:
            raise ValueError("MiniMax TTS requires a voice (voice_id)")
        self._model = model
        self._voice = voice

        api_key_secret = settings.minimax_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value().strip():
            raise ValueError("minimax_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"minimax-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with ws_client.connect(
                _WS_URL,
                additional_headers=headers,
                max_size=_MAX_WS_SIZE,
            ) as ws:
                async for message in ws:
                    if isinstance(message, (bytes, bytearray)):
                        continue
                    data: dict[str, Any] = json.loads(message)
                    event = data.get("event")
                    base = data.get("base_resp") or {}
                    if event == "task_failed" or base.get("status_code", 0) != 0:
                        raise RuntimeError(
                            f"{event}: [{base.get('status_code')}] {base.get('status_msg', '')}"
                        )
                    if event == "connected_success":
                        await ws.send(
                            json.dumps(
                                {
                                    "event": "task_start",
                                    "model": self._model,
                                    "voice_setting": {"voice_id": self._voice},
                                    "audio_setting": {
                                        "format": "pcm",
                                        "sample_rate": _SAMPLE_RATE,
                                        "channel": 1,
                                    },
                                }
                            )
                        )
                    elif event == "task_started":
                        start = time.monotonic()
                        await ws.send(json.dumps({"event": "task_continue", "text": text}))
                        await ws.send(json.dumps({"event": "task_finish"}))
                    elif event == "task_continued":
                        chunk = (data.get("data") or {}).get("audio")
                        if chunk:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(bytes.fromhex(chunk))
                        if data.get("is_final"):
                            break
                    elif event == "task_finished":
                        break

        except Exception as exc:
            logger.warning("minimax_tts_error", provider="minimax", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="minimax",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="minimax",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
