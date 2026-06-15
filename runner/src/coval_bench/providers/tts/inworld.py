# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Inworld AI TTS provider — WebSocket bidirectional streaming.

Wire protocol:
  connect wss://api.inworld.ai/tts/v1/voice:streamBidirectional
  → send JSON {"create": {voiceId, modelId, audioConfig}, "contextId": ...}
  → send JSON {"send_text": {"text": ..., "flush_context": {}}, "contextId": ...}
  → recv JSON chunks: {"result": {"audioChunk": {"audioContent": "<base64-pcm>"}}}
  → recv flush done:  {"result": {"flushCompleted": {}}}

Auth:   ?authorization=Basic <key> query param. INWORLD_API_KEY is already the
        portal-issued base64 credential, so it is sent verbatim (re-encoding fails auth).
Output: 16-bit mono PCM (base64-decoded from JSON envelopes), 24 kHz
"""

from __future__ import annotations

import base64
import json
import time
from urllib.parse import quote

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000
_WS_URL = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
_CONTEXT_ID = "coval-bench"


class InworldTTSProvider(TTSProvider):
    """Inworld AI TTS provider using WebSocket bidirectional streaming."""

    _VALID_MODELS = frozenset({"inworld-tts-1.5-max", "inworld-tts-1.5-mini"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Unsupported Inworld model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._model = model
        self._voice = voice

        api_key_secret = settings.inworld_api_key
        if api_key_secret is None:
            raise ValueError("inworld_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        # Model ids already carry the "inworld-tts-" prefix; don't double it.
        return self._model

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Inworld's WebSocket and return a TTSResult."""
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        url = f"{_WS_URL}?authorization={quote(f'Basic {self._api_key}', safe='')}"
        create_msg = json.dumps(
            {
                "create": {
                    "voiceId": self._voice,
                    "modelId": self._model,
                    "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": SAMPLE_RATE},
                },
                "contextId": _CONTEXT_ID,
            }
        )
        text_msg = json.dumps(
            {
                "send_text": {"text": text, "flush_context": {}},
                "contextId": _CONTEXT_ID,
            }
        )

        try:
            async with ws_client.connect(url) as ws:
                start = time.monotonic()
                await ws.send(create_msg)
                await ws.send(text_msg)

                async for raw in ws:
                    if isinstance(raw, bytes):
                        # Graceful fallback: accept raw binary if server ever sends it.
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(raw)
                        continue

                    msg = json.loads(raw)
                    if msg.get("error"):
                        return finalize_tts_result(
                            provider="inworld",
                            model=self._model,
                            voice=self._voice,
                            pcm=b"".join(audio_chunks),
                            sample_rate=SAMPLE_RATE,
                            audio_synthesis_start=start,
                            first_audio_chunk_at=first_chunk_at,
                            error=str(msg["error"])[:500],
                        )

                    result = msg.get("result", {})
                    audio_b64: str = result.get("audioChunk", {}).get("audioContent", "")
                    if audio_b64:
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(base64.b64decode(audio_b64))
                    elif "flushCompleted" in result:
                        break

        except Exception as exc:
            logger.warning("inworld_error", exc_info=True)
            return finalize_tts_result(
                provider="inworld",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="inworld",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
