# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Palabra real-time TTS streaming provider."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from uuid import uuid4

import httpx
import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("palabra-tts-v1",)
_VALID_VOICES = ("default_low", "default_high")
_SESSION_URL = "https://api.palabra.ai/session-storage/session"
_WS_URL = "wss://stream.us.palabra.ai/tts-api/v1/text-to-speech/stream"
_SAMPLE_RATE = 24000


class PalabraTTSProvider(TTSProvider):
    """Palabra TTS provider using WebSocket streaming (JSON frames, base64 audio)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Palabra TTS model {model!r}. Valid: {_VALID_MODELS}")
        if voice not in _VALID_VOICES:
            raise ValueError(f"Invalid Palabra TTS voice {voice!r}. Valid: {_VALID_VOICES}")
        self._model = model
        self._voice = voice

        api_key_secret = settings.palabra_api_key
        if api_key_secret is None:
            raise ValueError("palabra_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return "palabra"

    @property
    def model(self) -> str:
        return self._model

    async def _get_session_token(self) -> str:
        """POST /session-storage/session -> the session auth token."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                client_id, client_secret = self._api_key.split("_", 1)
            except ValueError as exc:
                raise ValueError("palabra_api_key must be '<client_id>_<client_secret>'") from exc
            resp = await client.post(
                _SESSION_URL,
                headers={"ClientID": client_id, "ClientSecret": client_secret},
                json={"data": {}},
            )
            resp.raise_for_status()
            token: str = resp.json()["data"]["publisher"]
        return token

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            # Pre-t0: get auth token
            token = await self._get_session_token()

            async with ws_client.connect(f"{_WS_URL}?token={token}") as ws:
                # Pre-t0: session init
                await ws.send(
                    json.dumps(
                        {
                            "type": "init",
                            "language": "en",
                            "model": self._model,
                            "voice_options": {"voice_id": self._voice},
                            "output": {"format": "pcm", "sample_rate": _SAMPLE_RATE},
                        }
                    )
                )

                start = time.monotonic()
                generation_id = f"coval_{uuid4().hex}"
                await ws.send(
                    json.dumps(
                        {
                            "type": "text",
                            "text": text,
                            "generation_id": generation_id,
                            "is_eos": True,
                        }
                    )
                )

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)

                    data: dict[str, Any] = event.get("data") or {}

                    if event.get("message_type") == "error":
                        raise RuntimeError(f"{data.get('code')}: {data.get('desc')}")

                    if event.get("message_type") != "audio_chunk":
                        continue

                    audio_b64: str = data.get("audio", "")
                    if audio_b64:
                        chunk = base64.b64decode(audio_b64)
                        if chunk:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(chunk)

                    if data.get("last_chunk"):
                        break

        except Exception as exc:
            logger.warning("palabra_tts_error", provider="palabra", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="palabra",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="palabra",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
