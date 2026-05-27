# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rime TTS provider — WebSocket streaming to Rime /ws3 JSON endpoint."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_WS_BASE = "wss://users-ws.rime.ai/ws3"

VALID_MODELS = {"arcana", "coda", "mistv3"}

# Native output sample rates per model; requesting each model's native rate
# avoids server-side upsampling and produces the most faithful benchmark audio.
_MODEL_SAMPLE_RATES: dict[str, int] = {
    "arcana": 24000,
    "coda": 24000,
    "mistv3": 22050,
}
if set(_MODEL_SAMPLE_RATES) != VALID_MODELS:
    raise ValueError(
        f"VALID_MODELS and _MODEL_SAMPLE_RATES are out of sync: "
        f"{VALID_MODELS ^ set(_MODEL_SAMPLE_RATES)}"
    )


class RimeTTSProvider(TTSProvider):
    """Rime TTS provider using WebSocket /ws3 JSON streaming."""

    enabled: bool = False  # enabled via DEFAULT_TTS_MATRIX entries

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

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
        if self._model not in VALID_MODELS:
            return TTSResult(
                provider="rime",
                model=self._model,
                voice=self._voice,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported Rime model: {self._model}. Valid models: {sorted(VALID_MODELS)}"
                ),
            )

        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None
        sample_rate = _MODEL_SAMPLE_RATES[self._model]

        qs = urlencode(
            {
                "modelId": self._model,
                "speaker": self._voice or "luna",
                "audioFormat": "pcm",
                "samplingRate": sample_rate,
                # segment=never: synthesis fires only on explicit eos, not on sentence
                "segment": "never",
            }
        )
        url = f"{_WS_BASE}?{qs}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with ws_client.connect(url, additional_headers=headers) as ws:
                start = time.monotonic()

                await ws.send(json.dumps({"text": text}))
                await ws.send(json.dumps({"operation": "eos"}))

                async for raw in ws:
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "chunk":
                        audio_bytes = base64.b64decode(msg["data"])
                        if audio_bytes:
                            if ttfa_ms is None:
                                ttfa_ms = (time.monotonic() - start) * 1000
                                logger.debug(
                                    "rime_ttfa",
                                    model=self._model,
                                    ttfa_ms=ttfa_ms,
                                )
                            audio_chunks.append(audio_bytes)

                    elif msg_type == "done":
                        break

                    elif msg_type == "error":
                        raise RuntimeError(msg.get("message", "rime /ws3 error"))
                    # "timestamps" events are silently dropped — not needed for benchmark.

        except Exception as exc:
            logger.warning("rime_error", exc_info=True)
            return TTSResult(
                provider="rime",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, sample_rate) if audio_chunks else None
        return TTSResult(
            provider="rime",
            model=self._model,
            voice=self._voice,
            ttfa_ms=ttfa_ms,
            audio_path=audio_path,
            error=None,
        )


def _write_wav(chunks: list[bytes], sample_rate: int) -> Path:
    """Concatenate raw PCM chunks and write a WAV file to a temp location."""
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_data = b"".join(chunks)
    with wave.open(tmp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return Path(tmp_name)
