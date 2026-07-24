# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fluxions hosted VUI TTS provider (WebSocket streaming).

Wire protocol on wss://api.fluxions.ai/vui/v1/tts/ws:
  connect → send speak(voice, input) → recv {"type": "start"}
  → recv binary s16le PCM @ 24 kHz frames → recv {"type": "done"}

Built-in voices are public: the render path takes no credential, so there is no
API key in ``Settings`` for this provider and nothing for infra to mount. The
request carries no model identifier either, only a voice, so the registry entry
is the bare surface name ``vui``.

``verify_chunks`` is sent as ``False``. The server default (``True``) re-checks
every rendered chunk with an STT pass and re-renders any that misread the text.
Measured over two 10-prompt A/B samples on ``tts-v1`` it raised median TTFA from
~295 ms to ~1.4 s (worst row 8.5 s) without improving WER, so the unverified
stream is both the faster and the comparable configuration.

Voice ids are ``<name>.<catalog-hash>`` and the hash rotates when Fluxions
republishes the catalog, so the registry pins the stable bare name and this
module resolves it against ``GET /vui/voices``. Resolution is cached process-wide
and always happens before the TTFA clock starts.
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

# name → full "<name>.<hash>" voice id, from GET /vui/voices. Process-wide: the
# catalog is public and identical for every model entry in a run.
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
        """Map the registry's bare voice name to the catalog's hashed voice id.

        A voice already carrying a hash suffix passes through untouched, so a
        fully-qualified id in the registry still works. An unknown name also
        passes through, letting the server reject it with a real error rather
        than guessing a substitute voice.
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
            async with ws_client.connect(_WS_URL) as ws:
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
