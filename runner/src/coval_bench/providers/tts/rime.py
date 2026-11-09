# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rime TTS provider — WebSocket streaming (Coda v1 protocol, legacy /ws3 for the rest)."""

from __future__ import annotations

import base64
import json
import time
from urllib.parse import urlencode

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

_CODA_WS_URL = "wss://coda.api.rime.ai/ws"
_LEGACY_WS_BASE = "wss://users-ws.rime.ai/ws3"

_MODEL_SAMPLE_RATES: dict[str, int] = {
    "coda": 24000,
    "mistv3": 22050,
}


class RimeTTSProvider(TTSProvider):
    """Rime TTS provider.

    Coda uses its own host (coda.api.rime.ai, v1 JSON protocol). mistv3 stays
    on the legacy users-ws.rime.ai /ws3 endpoint.
    """

    def __init__(self, settings: Settings, model: str, voice: str | None) -> None:
        self._model = model
        self._voice = voice or "luna"

        api_key_secret = settings.rime_api_key
        if api_key_secret is None:
            raise ValueError("rime_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"rime-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech over WebSocket and return a TTSResult."""
        sample_rate = _MODEL_SAMPLE_RATES.get(self._model, 24000)
        synthesis = Synthesis("rime", self._model, self._voice, sample_rate)
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            if self._model == "coda":
                await self._synthesize_coda(text, sample_rate, headers, synthesis)
            else:
                await self._synthesize_legacy(text, sample_rate, headers, synthesis)
        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()

    async def _synthesize_coda(
        self, text: str, sample_rate: int, headers: dict[str, str], synthesis: Synthesis
    ) -> None:
        async with ws_client.connect(_CODA_WS_URL, additional_headers=headers) as ws:
            ready = json.loads(await ws.recv())
            if "ready" not in ready:
                raise RuntimeError(f"rime coda: expected ready, got {ready}")

            synthesis.start = time.monotonic()
            await ws.send(
                json.dumps(
                    {
                        "contextId": "turn-1",
                        "start": {
                            "speaker": self._voice,
                            "language": "en",
                            "text": text,
                            "audioParameters": {
                                "audioFormat": "audio/pcm",
                                "samplingRate": sample_rate,
                            },
                        },
                    }
                )
            )

            async for raw in ws:
                msg = json.loads(raw)
                if "audio" in msg:
                    synthesis.add_chunk(base64.b64decode(msg["audio"]))
                elif "done" in msg or "cancelled" in msg:
                    break
                elif "error" in msg:
                    raise RuntimeError(msg["error"].get("message", "rime coda error"))

    async def _synthesize_legacy(
        self, text: str, sample_rate: int, headers: dict[str, str], synthesis: Synthesis
    ) -> None:
        qs = urlencode(
            {
                "modelId": self._model,
                "speaker": self._voice,
                "audioFormat": "pcm",
                "samplingRate": sample_rate,
                # segment=never: synthesis fires only on explicit eos, not on sentence
                "segment": "never",
            }
        )
        url = f"{_LEGACY_WS_BASE}?{qs}"

        async with ws_client.connect(url, additional_headers=headers) as ws:
            synthesis.start = time.monotonic()

            await ws.send(json.dumps({"text": text}))
            await ws.send(json.dumps({"operation": "eos"}))

            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "chunk":
                    synthesis.add_chunk(base64.b64decode(msg["data"]))
                elif msg_type == "done":
                    break
                elif msg_type == "error":
                    raise RuntimeError(msg.get("message", "rime /ws3 error"))
