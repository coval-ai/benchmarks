# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fluxions hosted VUI TTS provider (WebSocket streaming).

Wire protocol on wss://api.fluxions.ai/vui/v1/tts/ws:
  connect → send speak(voice, input) → recv {"type": "start"}
  → recv binary s16le PCM @ 24 kHz frames → recv {"type": "done"}

The websocket requires an API key, sent as a bearer ``Authorization`` header
on the handshake; unauthenticated connects are closed 1008 by a bot gate. The
voice catalog endpoint is public. ``verify_chunks`` is disabled: the server's
STT re-render pass multiplies TTFA without improving WER. Voice ids are
``<name>.<catalog-hash>`` with a rotating hash, so the registry pins the bare
name and this module resolves it against ``GET /vui/voices`` before t0.
"""

from __future__ import annotations

import json
import time

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers._http_session import get_shared_client
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("vui",)
_BASE_URL = "https://api.fluxions.ai"
_WS_URL = "wss://api.fluxions.ai/vui/v1/tts/ws"
_SAMPLE_RATE = 24000

# name → "<name>.<hash>" catalog id, shared process-wide
_VOICE_IDS: dict[str, str] = {}


async def _load_voice_catalog() -> dict[str, str]:
    """Fetch ``GET /vui/voices`` and index the ids by their bare name."""
    client = get_shared_client("fluxions", _BASE_URL)
    response = await client.get("/vui/voices")
    response.raise_for_status()
    voices = response.json().get("voices") or []
    return {
        str(v["voice_id"]).split(".", 1)[0]: str(v["voice_id"]) for v in voices if v.get("voice_id")
    }


class FluxionsTTSProvider(TTSProvider):
    """Fluxions VUI TTS provider using WebSocket streaming (binary PCM frames)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Fluxions TTS model {model!r}. Valid: {_VALID_MODELS}")
        if not voice:
            raise ValueError("Fluxions TTS requires a voice")
        api_key_secret = settings.fluxions_api_key
        if api_key_secret is None:
            raise ValueError("fluxions_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()
        self._model = model
        self._voice = voice

    @property
    def name(self) -> str:
        return f"fluxions-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Cache the voice catalog before t0 so no render pays for the lookup."""
        t0 = time.monotonic()
        _VOICE_IDS.update(await _load_voice_catalog())
        logger.info(
            "fluxions_prewarm",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            voices=len(_VOICE_IDS),
        )

    async def _resolve_voice(self) -> str:
        """Map the bare registry voice name to the hashed catalog id.

        Hashed ids and unknown names pass through so the server reports
        the real error instead of us guessing a substitute.
        """
        if "." in self._voice:
            return self._voice
        if self._voice not in _VOICE_IDS:
            _VOICE_IDS.update(await _load_voice_catalog())
        resolved = _VOICE_IDS.get(self._voice)
        if resolved is None:
            logger.warning(
                "fluxions_voice_unresolved",
                provider="fluxions",
                model=self._model,
                voice=self._voice,
            )
            return self._voice
        return resolved

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            voice_id = await self._resolve_voice()
            async with ws_client.connect(
                _WS_URL, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            ) as ws:
                start = time.monotonic()
                await ws.send(
                    json.dumps(
                        {
                            "type": "speak",
                            "voice": voice_id,
                            "input": text,
                            "verify_chunks": False,
                        }
                    )
                )

                async for message in ws:
                    if isinstance(message, (bytes, bytearray)):
                        if message:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(bytes(message))
                        continue

                    event = json.loads(message)
                    if event.get("type") == "error":
                        raise RuntimeError(event.get("message", "unknown error"))
                    if event.get("type") == "done":
                        break

        except Exception as exc:
            logger.warning(
                "fluxions_tts_error", provider="fluxions", model=self._model, exc_info=exc
            )
            return finalize_tts_result(
                provider="fluxions",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="fluxions",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
