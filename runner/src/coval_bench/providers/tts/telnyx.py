# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Telnyx TTS provider.

Telnyx's own engines (Ultra, Qwen3TTS, KokoroTTS, Bayan, Sukhan) and the
third-party vendors it fronts share one voice namespace,
``<Provider>.<Model>.<voice_id>``, which here is ``model`` plus ``voice``.
Two transports, picked per model:

HTTP  POST https://api.telnyx.com/v2/text-to-speech/speech, chunked body, so
      TTFA is the first body chunk. Ultra is HTTP-only (the socket answers
      403) and the third-party vendors return PCM here.
WS    wss://api.telnyx.com/v2/text-to-speech/speech?voice=..&audio_format=pcm
      Handshake ``{"text": " "}``, then the text, then ``{"text": ""}``;
      audio arrives as base64 in JSON frames until ``isFinal``. Qwen3TTS
      only honours the PCM request on this path.

KokoroTTS returns MP3 on both paths whatever format is asked for, so it is
not benchmarkable here; the MP3 guard fails such a run instead of scoring
compressed bytes as PCM. The audio cache is bypassed on every request so a
repeated prompt is synthesized again rather than replayed.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers._http_session import (
    connection_reused,
    get_shared_client,
    submit_to_headers_ms,
)
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000

_BASE_URL = "https://api.telnyx.com"
_SPEECH_PATH = "/v2/text-to-speech/speech"
_WS_URL = "wss://api.telnyx.com/v2/text-to-speech/speech"

# Each vendor spells "raw 24 kHz PCM" differently in its provider params block.
_PCM_PARAMS: dict[str, dict[str, Any]] = {
    "telnyx": {"response_format": "pcm", "sampling_rate": SAMPLE_RATE},
    "assemblyai": {"audio_format": "pcm_s16le", "sample_rate": SAMPLE_RATE},
    "soniox": {"audio_format": "pcm_s16le", "sample_rate": SAMPLE_RATE},
    "xai": {"output_format": "pcm", "sample_rate": SAMPLE_RATE},
}

# Models whose HTTP response is MP3 no matter what, but stream PCM on the socket.
_WS_MODELS = frozenset({"Telnyx.Qwen3TTS"})

_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
_MP3_ERROR = "provider returned MP3 instead of the requested PCM"


def vendor_of(model: str) -> str:
    """The provider segment of a ``<Provider>.<Model>`` id, lowercased for the params key."""
    vendor = model.split(".", 1)[0].lower()
    if vendor not in _PCM_PARAMS:
        raise ValueError(
            f"Telnyx TTS vendor {vendor!r} (from model {model!r}) has no PCM output mapping. "
            f"Known: {sorted(_PCM_PARAMS)}"
        )
    return vendor


def _looks_like_mp3(chunk: bytes) -> bool:
    return chunk.startswith(_MP3_MAGIC)


class TelnyxTTSProvider(TTSProvider):
    """Telnyx TTS, PCM out over HTTP or the streaming socket."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._vendor = vendor_of(model)
        self._model = model
        self._voice = voice

        api_key_secret = settings.telnyx_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value():
            raise ValueError("telnyx_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"telnyx-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def full_voice(self) -> str:
        return f"{self._model}.{self._voice}"

    @property
    def uses_websocket(self) -> bool:
        return self._model in _WS_MODELS

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Pre-warm the shared httpx pool; a 401 on the HEAD still warms the socket."""
        client = get_shared_client("telnyx", _BASE_URL)
        t0 = time.monotonic()
        response = await client.head(_SPEECH_PATH)
        logger.info(
            "telnyx_tts_prewarm",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            http_version=response.http_version,
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "text": text,
            "voice": self.full_voice,
            "output_type": "binary_output",
            "disable_cache": True,
            self._vendor: _PCM_PARAMS[self._vendor],
        }

    def _ws_url(self) -> str:
        return (
            f"{_WS_URL}?voice={self.full_voice}"
            f"&audio_format=pcm&sample_rate={SAMPLE_RATE}&disable_cache=true"
        )

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("telnyx", self._model, self._voice, SAMPLE_RATE)
        try:
            if self.uses_websocket:
                await self._synthesize_ws(text, synthesis)
            else:
                await self._synthesize_http(text, synthesis)
        except Exception as exc:
            synthesis.fail(exc)
        return synthesis.result()

    async def _synthesize_http(self, text: str, synthesis: Synthesis) -> None:
        client = get_shared_client("telnyx", _BASE_URL)
        headers = {**self._headers(), "Content-Type": "application/json"}
        synthesis.start = time.monotonic()
        async with client.stream(
            "POST", _SPEECH_PATH, headers=headers, json=self._payload(text)
        ) as response:
            synthesis.http_version = response.http_version
            synthesis.submit_to_headers_ms = submit_to_headers_ms(response.request)
            synthesis.connection_reused = connection_reused(response.request)
            if response.is_error:
                body = await response.aread()
                detail = body.decode("utf-8", "replace").strip() or response.reason_phrase
                synthesis.start = None
                synthesis.error = f"HTTP {response.status_code}: {detail[:500]}"
                synthesis.status_code = response.status_code
                return
            if response.headers.get("content-type", "").startswith("audio/mpeg"):
                raise RuntimeError(_MP3_ERROR)
            async for chunk in response.aiter_bytes():
                synthesis.add_chunk(chunk)

    async def _synthesize_ws(self, text: str, synthesis: Synthesis) -> None:
        async with ws_client.connect(self._ws_url(), additional_headers=self._headers()) as ws:
            await ws.send(json.dumps({"text": " "}))
            synthesis.start = time.monotonic()
            await ws.send(json.dumps({"text": text}))
            await ws.send(json.dumps({"text": ""}))

            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                event: dict[str, Any] = json.loads(raw)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                audio = event.get("audio")
                if audio:
                    chunk = base64.b64decode(audio)
                    if not synthesis.chunks and _looks_like_mp3(chunk):
                        raise RuntimeError(_MP3_ERROR)
                    synthesis.add_chunk(chunk)
                else:
                    synthesis.keep_frame(raw)
                if event.get("isFinal"):
                    break
