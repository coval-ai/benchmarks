# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Inworld AI TTS provider — WebSocket bidirectional streaming.

INWORLD_API_KEY is the portal-issued base64 credential and rides the
`?authorization=Basic <key>` query param verbatim; re-encoding it fails auth.
"""

from __future__ import annotations

import base64
import json
import time
from urllib.parse import quote

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

SAMPLE_RATE = 24000
_WS_URL = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
_CONTEXT_ID = "coval-bench"


def _pcm_from_chunk(chunk: bytes) -> bytes:
    """Unwrap Inworld's per-chunk LINEAR16 WAV container, returning raw PCM.

    Each audio chunk is a standalone WAV (RIFF header + ``data`` subchunk); the
    pipeline assembles headerless PCM, so the header is stripped here. Non-WAV
    chunks pass through unchanged.
    """
    if chunk[:4] != b"RIFF":
        return chunk
    data = chunk.find(b"data")
    return chunk[data + 8 :] if data != -1 else chunk


class InworldTTSProvider(TTSProvider):
    """Inworld AI TTS provider using WebSocket bidirectional streaming."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
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
        synthesis = Synthesis("inworld", self._model, self._voice, SAMPLE_RATE)

        auth_param = quote(f"Basic {self._api_key}", safe="")
        url = f"{_WS_URL}?authorization={auth_param}"
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
                synthesis.start = time.monotonic()
                await ws.send(create_msg)
                await ws.send(text_msg)

                async for raw in ws:
                    if isinstance(raw, bytes):
                        # Graceful fallback: accept raw binary if server ever sends it.
                        synthesis.add_chunk(_pcm_from_chunk(raw))
                        continue

                    msg = json.loads(raw)
                    if msg.get("error"):
                        synthesis.error = str(msg["error"])[:500]
                        return synthesis.result()

                    result = msg.get("result", {})
                    audio_b64: str = result.get("audioChunk", {}).get("audioContent", "")
                    if audio_b64:
                        synthesis.add_chunk(_pcm_from_chunk(base64.b64decode(audio_b64)))
                    elif "flushCompleted" in result:
                        break

        except Exception as exc:
            # Credential rides in the URL query; scrub it before the exception
            # reaches logs or the stored result. No exc_info — the rendered
            # traceback would echo the unredacted URI.
            scrubbed = str(exc).replace(auth_param, "***")
            synthesis.fail(exc, exc_info=False, error=scrubbed)
            synthesis.error = scrubbed

        return synthesis.result()
