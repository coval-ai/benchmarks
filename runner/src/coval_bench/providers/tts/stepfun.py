# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""StepFun streaming TTS provider (WebSocket, JSON frames, base64 PCM).

The socket opens a session (``tts.connection.done`` then ``tts.create`` /
``tts.response.created``) before any text is sent; t0 is the text submission.
``mode: sentence`` because the prompt is complete up front: the default mode
buffers for LLM token streams and only starts once a full sentence accumulates.
Protocol: https://platform.stepfun.ai/docs/en/api-reference/audio/ws-audio
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

_WS_URL = "wss://api.stepfun.ai/v1/realtime/audio"
_SAMPLE_RATE = 24000


class StepfunTTSProvider(TTSProvider):
    """Synthesize English benchmark prompts with StepFun's StepAudio TTS models."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        api_key_secret = settings.stepfun_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value():
            raise ValueError("stepfun_api_key is required in Settings")
        self._model = model
        self._voice = voice
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"stepfun-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            headers = {"Authorization": f"Bearer {self._api_key}"}
            async with ws_client.connect(
                f"{_WS_URL}?model={self._model}", additional_headers=headers
            ) as ws:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)
                    event_type = str(event.get("type", ""))
                    data: dict[str, Any] = event.get("data") or {}

                    if event_type == "tts.connection.done":
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "tts.create",
                                    "data": {
                                        "session_id": data["session_id"],
                                        "voice_id": self._voice,
                                        "language": "en",
                                        "response_format": "pcm",
                                        "sample_rate": _SAMPLE_RATE,
                                        "mode": "sentence",
                                    },
                                }
                            )
                        )
                    elif event_type == "tts.response.created":
                        session_id = data["session_id"]
                        start = time.monotonic()
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "tts.text.delta",
                                    "data": {"session_id": session_id, "text": text},
                                }
                            )
                        )
                        await ws.send(
                            json.dumps(
                                {"type": "tts.text.done", "data": {"session_id": session_id}}
                            )
                        )
                    elif event_type == "tts.response.audio.delta":
                        chunk = base64.b64decode(data.get("audio") or "")
                        if chunk:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(chunk)
                    elif event_type == "tts.response.audio.done":
                        break
                    elif event_type == "tts.response.error":
                        message = data.get("message") or "StepFun TTS error"
                        code = data.get("code")
                        raise RuntimeError(f"{message} (code={code})" if code else str(message))

        except Exception as exc:
            logger.warning("stepfun_tts_error", provider="stepfun", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="stepfun",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc) or type(exc).__name__,
            )

        return finalize_tts_result(
            provider="stepfun",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
