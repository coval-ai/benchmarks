# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepdub TTS provider — the streaming in/out WebSocket API.

Wire protocol on wss://wss.deepdub.ai/ws (``x-api-key`` header at the
handshake): recv the ``status``/``connected`` welcome → send ``stream-config``
→ ``stream-text`` → ``end-stream`` → recv JSON frames whose ``data`` field
carries base64 PCM until the frame with ``isFinal``, which may itself carry
audio. Session errors arrive as ``action: error`` with ``message``; worker
errors as frames with ``error``.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

_WS_URL = "wss://wss.deepdub.ai/ws"
_LOCALE = "en-US"

SAMPLE_RATE = 24000


def _raise_on_error(frame: dict[str, Any]) -> None:
    """Raise on a session or worker error frame; others pass."""
    if frame.get("action") == "error":
        raise RuntimeError(f"deepdub session error: {frame.get('message')}")
    if frame.get("error") is not None:
        raise RuntimeError(f"deepdub worker error: {frame['error']}")


class DeepdubTTSProvider(TTSProvider):
    """Deepdub TTS provider using the streaming WebSocket API (base64 PCM frames)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if not voice:
            raise ValueError("Deepdub TTS requires a voice prompt id")
        self._model = model
        self._voice = voice

        api_key_secret = settings.deepdub_api_key
        if api_key_secret is None:
            raise ValueError("deepdub_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"deepdub-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("deepdub", self._model, self._voice, SAMPLE_RATE)
        finished = False

        config = {
            "action": "stream-config",
            "data": {
                "model": self._model,
                "voicePromptId": self._voice,
                "locale": _LOCALE,
                "format": "s16le",
                "sampleRate": SAMPLE_RATE,
                "realtime": True,
            },
        }

        try:
            async with ws_client.connect(
                _WS_URL, additional_headers={"x-api-key": self._api_key}
            ) as ws:
                welcome = json.loads(await ws.recv())
                if welcome.get("message") != "connected":
                    raise RuntimeError(f"deepdub handshake failed: {welcome}")
                await ws.send(json.dumps(config))

                synthesis.start = time.monotonic()
                await ws.send(json.dumps({"action": "stream-text", "data": {"text": text}}))
                await ws.send(json.dumps({"action": "end-stream"}))

                async for raw in ws:
                    text_frame = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                    try:
                        frame = json.loads(text_frame)
                    except json.JSONDecodeError:
                        synthesis.keep_frame(text_frame)
                        continue

                    _raise_on_error(frame)
                    data = frame.get("data")
                    pcm = base64.b64decode(data) if data else b""
                    if pcm:
                        synthesis.add_chunk(pcm)
                    else:
                        synthesis.keep_frame(text_frame)
                    if frame.get("isFinal"):
                        finished = True
                        break

                # A clean close before ``isFinal`` is a truncated stream; audio
                # collected so far must not be scored as a complete synthesis.
                if not finished:
                    raise RuntimeError("connection closed before the isFinal frame")

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
