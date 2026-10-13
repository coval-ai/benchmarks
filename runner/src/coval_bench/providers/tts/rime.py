# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rime TTS provider — WebSocket streaming to Rime /ws3 JSON endpoint."""

from __future__ import annotations

import base64
import json
import time
from urllib.parse import urlencode

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

_WS_BASE = "wss://users-ws.rime.ai/ws3"

_MODEL_SAMPLE_RATES: dict[str, int] = {
    "arcana": 24000,
    "coda": 24000,
    "mistv3": 22050,
}


class RimeTTSProvider(TTSProvider):
    """Rime TTS provider using WebSocket /ws3 JSON streaming."""

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
        """Synthesize speech via Rime /ws3 WebSocket and return a TTSResult."""
        sample_rate = _MODEL_SAMPLE_RATES.get(self._model, 24000)
        synthesis = Synthesis("rime", self._model, self._voice, sample_rate)

        qs = urlencode(
            {
                "modelId": self._model,
                "speaker": self._voice or "luna",
                "audioFormat": "pcm",
                "samplingRate": _MODEL_SAMPLE_RATES.get(self._model, 24000),
                # segment=never: synthesis fires only on explicit eos, not on sentence
                "segment": "never",
            }
        )
        url = f"{_WS_BASE}?{qs}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
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
                    # "timestamps" events are silently dropped — not needed for benchmark.

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
