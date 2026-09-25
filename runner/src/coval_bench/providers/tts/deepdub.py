# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepdub TTS provider — WebSocket streaming to the eTTS realtime API.

Wire protocol on wss://wsapi.deepdub.ai/open (``x-api-key`` header at the
handshake): send one ``text-to-speech`` request → recv JSON frames whose
``data`` field carries base64 PCM (an empty-data ack precedes audio) → the
final frame sets ``isFinished`` and may itself carry audio. Errors arrive as
frames with ``error``/``errorType``.
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

_WS_URL = "wss://wsapi.deepdub.ai/open"
_LOCALE = "en-US"

SAMPLE_RATE = 24000


def _raise_on_error(frame: dict[str, Any]) -> None:
    """Raise on a server error frame (``error``/``errorType``); others pass."""
    if frame.get("error") is not None:
        error_type = frame.get("errorType") or "error"
        raise RuntimeError(f"deepdub {error_type}: {frame['error']}")


class DeepdubTTSProvider(TTSProvider):
    """Deepdub TTS provider using the realtime WebSocket API (base64 PCM frames)."""

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

        request = {
            "action": "text-to-speech",
            "model": self._model,
            "targetText": text,
            "locale": _LOCALE,
            "voicePromptId": self._voice,
            "format": "s16le",
            "sampleRate": SAMPLE_RATE,
            "realtime": True,
        }

        try:
            async with ws_client.connect(
                _WS_URL, additional_headers={"x-api-key": self._api_key}
            ) as ws:
                synthesis.start = time.monotonic()
                await ws.send(json.dumps(request))

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
                    if frame.get("isFinished"):
                        finished = True
                        break

                # A clean close before ``isFinished`` is a truncated stream; audio
                # collected so far must not be scored as a complete synthesis.
                if not finished:
                    raise RuntimeError("connection closed before the isFinished frame")

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
