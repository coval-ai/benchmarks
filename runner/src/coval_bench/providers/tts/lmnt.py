# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""LMNT TTS provider — WebSocket streaming via the speech sessions API."""

from __future__ import annotations

import json
import time
from typing import Any

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

_WS_URL = "wss://api.lmnt.com/v1/ai/speech/stream"
_LMNT_VERSION = "1.1"

SAMPLE_RATE = 24000


class LmntTTSProvider(TTSProvider):
    """LMNT TTS provider using the speech sessions WebSocket API."""

    def __init__(self, settings: Settings, model: str, voice: str | None) -> None:
        self._model = model
        self._voice = voice or "leah"

        api_key_secret = settings.lmnt_api_key
        if api_key_secret is None:
            raise ValueError("lmnt_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"lmnt-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via an LMNT speech session and return a TTSResult."""
        synthesis = Synthesis("lmnt", self._model, self._voice, SAMPLE_RATE)

        init_msg = {
            "type": "init",
            "X-API-Key": self._api_key,
            "lmnt-version": _LMNT_VERSION,
            "voice": self._voice,
            "format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
            "language": "en",
        }

        try:
            async with ws_client.connect(_WS_URL) as ws:
                await ws.send(json.dumps(init_msg))
                # Wait for the server's `ready` ack so session setup stays out of TTFA.
                _check_error(json.loads(await ws.recv()))

                synthesis.start = time.monotonic()
                await ws.send(json.dumps({"type": "text", "text": text}))
                await ws.send(json.dumps({"type": "finish"}))

                async for msg in ws:
                    if isinstance(msg, bytes):
                        synthesis.add_chunk(msg)
                    else:
                        _check_error(json.loads(msg))

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()


def _check_error(payload: dict[str, Any]) -> None:
    """Raise on a server `error` envelope; any other JSON message is ignored."""
    if payload.get("type") == "error":
        raise RuntimeError(f"lmnt session error: {payload.get('error')}")
