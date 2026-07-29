# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""LMNT TTS provider — WebSocket streaming via the speech sessions API."""

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

_WS_URL = "wss://api.lmnt.com/v1/ai/speech/stream"
_LMNT_VERSION = "1.1"

SAMPLE_RATE = 24000


class LmntTTSProvider(TTSProvider):
    """LMNT TTS provider using the speech sessions WebSocket API."""

    # Speech sessions take no model parameter — every session runs Blizzard,
    # LMNT's only model. The id exists for registry identity and naming.
    _VALID_MODELS = frozenset({"blizzard"})

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
        if not self._model_supported(self._model):
            return TTSResult(
                provider="lmnt",
                model=self._model,
                voice=self._voice,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported LMNT model: {self._model}. "
                    f"Valid models: {sorted(self._VALID_MODELS)}"
                ),
            )

        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

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
                # The server acks the init with a `ready` message. Waiting for it
                # keeps session setup (auth, voice load) out of TTFA: sessions are
                # long-lived in real use, with text streamed into an established
                # session.
                _check_error(json.loads(await ws.recv()))

                start = time.monotonic()
                await ws.send(json.dumps({"type": "text", "text": text}))
                # `finish` flushes the buffered text and closes the session once
                # all audio has been dispatched, ending the receive loop.
                await ws.send(json.dumps({"type": "finish"}))

                async for msg in ws:
                    if isinstance(msg, bytes):
                        if msg:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(msg)
                    else:
                        _check_error(json.loads(msg))

        except Exception as exc:
            logger.warning("lmnt_error", provider="lmnt", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="lmnt",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="lmnt",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )


def _check_error(payload: dict[str, Any]) -> None:
    """Raise on a server `error` envelope; any other JSON message is ignored."""
    if payload.get("type") == "error":
        raise RuntimeError(f"lmnt session error: {payload.get('error')}")
