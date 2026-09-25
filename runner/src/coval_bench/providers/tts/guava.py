# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Guava TTS provider (daytona-tts).

Wire protocol: HTTP POST, <guava_base_url>/audio/speech, streaming response.
Auth: Authorization: Bearer <key>.
Request (JSON): {"input": text, "voice": ..., "sampling_rate": 16000,
  "response_format": "wav", "stream": true}
Response: raw 16 kHz mono PCM (audio/L16), or WAV; timing starts at PCM arrival.
"""

from __future__ import annotations

import io
import time
import wave

import structlog

from coval_bench.config import Settings
from coval_bench.providers._http_session import (
    connection_reused,
    get_shared_client,
    submit_to_headers_ms,
)
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 16000

_ENDPOINT = "/audio/speech"


class GuavaTTSProvider(TTSProvider):
    """Guava TTS provider."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        if not settings.guava_base_url:
            raise ValueError("guava_base_url is required in Settings")
        self._base_url = settings.guava_base_url

        api_key_secret = settings.guava_api_key
        if api_key_secret is None:
            raise ValueError("guava_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"guava-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Wake and warm the self-hosted endpoint; deletes its artifact; never fatal."""
        if not settings.guava_base_url or settings.guava_api_key is None:
            return
        provider = cls(settings, "daytona-tts", "grace")
        t0 = time.monotonic()
        result = await provider.synthesize("Warm up.")
        if result.audio_path is not None:
            result.audio_path.unlink(missing_ok=True)
        logger.info(
            "guava_prewarm",
            provider="guava",
            model="daytona-tts",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            error=result.error,
        )

    async def synthesize(self, text: str) -> TTSResult:
        client = get_shared_client("guava", self._base_url)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": text,
            "voice": self._voice,
            "sampling_rate": SAMPLE_RATE,
            "response_format": "wav",
            "stream": True,
        }

        synthesis = Synthesis("guava", self._model, self._voice, SAMPLE_RATE)
        arrivals: list[tuple[int, float]] = []
        received_bytes = 0

        try:
            synthesis.start = time.monotonic()
            async with client.stream("POST", _ENDPOINT, headers=headers, json=payload) as response:
                synthesis.http_version = response.http_version
                synthesis.submit_to_headers_ms = submit_to_headers_ms(response.request)
                synthesis.connection_reused = connection_reused(response.request)
                if response.is_error:
                    body = await response.aread()
                    detail = body.decode("utf-8", "replace").strip() or response.reason_phrase
                    synthesis.start = None
                    synthesis.error = f"HTTP {response.status_code}: {detail[:500]}"
                    synthesis.status_code = response.status_code
                    return synthesis.result()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        received_bytes += len(chunk)
                        arrivals.append((received_bytes, time.monotonic()))
                        synthesis.chunks.append(chunk)
            if synthesis.chunks:
                payload_audio = b"".join(synthesis.chunks)
                pcm_offset = 0
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if payload_audio[:4] == b"RIFF" or content_type != "audio/l16":
                    source = io.BytesIO(payload_audio)
                    with wave.open(source, "rb") as wav:
                        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
                            1,
                            2,
                            SAMPLE_RATE,
                        ):
                            raise ValueError("Guava requires mono 16-bit 16 kHz WAV output")
                        pcm_offset = source.tell()
                        audio_data = wav.readframes(wav.getnframes())
                else:
                    audio_data = payload_audio
                synthesis.chunks = [audio_data]
                if audio_data:
                    synthesis.first_chunk_at = next(at for end, at in arrivals if end > pcm_offset)
        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
