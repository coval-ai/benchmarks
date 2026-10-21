# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseten dedicated-endpoint STT provider (Whisper Large V3, Qwen3-ASR).

Baseten fronts several dedicated deployments under one provider name, and they
do not all speak the same wire protocol:

``binary``
    Whisper deployments. Raw PCM as binary frames of exactly 512 samples
    (1024 bytes) — the Silero VAD frame size; other frame sizes are not
    processed correctly — finalized with ``{"type": "end_audio"}``. The server
    answers ``end_audio`` twice ("acknowledged", then "finished") and never
    closes the session itself.

``buffer``
    Qwen3-ASR deployments. Binary frames are rejected; PCM goes base64-encoded
    inside ``input_audio_buffer.append`` messages, finalized with
    ``input_audio_buffer.commit``. The last transcription carries
    ``is_end_of_audio_flush``; no ``end_audio`` message is ever sent.

Both protocols open with the same session frame and return the same
``transcription`` message shape. Endpoint URLs embed private, pre-launch model
ids, so they are injected from settings rather than hardcoded.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

# Silero VAD frame: 512 samples * 2 bytes = 1024 bytes; other sizes are rejected.
_FRAME_SAMPLES = 512
_FRAME_BYTES = _FRAME_SAMPLES * 2
_PCM_RATE = 16000
_APPEND_BYTES = int(_PCM_RATE * 0.1) * 2
_MAX_WS_SIZE = 16 * 1024 * 1024
# Measurement handshake cap; warmup connects with its own budget-length timeout.
_OPEN_TIMEOUT_S = 45
# Identical across endpoints so latency deltas are model, not configuration.
_PARTIAL_INTERVAL_S = 0.3
_WARMUP_SECONDS = 1.0
# A measurement failure; warmup maps it to success (the tone has no speech).
_NO_FINAL_ERROR = "Baseten stream ended before a final transcription was received"

# The warmup is the scale-up trigger; must outlast a boot (~253 s observed).
_WARMUP_TIMEOUT_S = 360


class _Protocol(StrEnum):
    """How a deployment wants its audio framed."""

    BINARY = "binary"
    BUFFER = "buffer"


@dataclass(frozen=True)
class _Endpoint:
    """Where a model is deployed and how to talk to it."""

    url_attr: str  # Settings attribute holding the wss:// URL
    protocol: _Protocol


_ENDPOINTS: dict[str, _Endpoint] = {
    "whisper-large-v3": _Endpoint("baseten_whisper_url", _Protocol.BINARY),
    "qwen3-asr-1.7b": _Endpoint("baseten_qwen_asr_url", _Protocol.BUFFER),
}


def endpoint_url(settings: Settings, model: str) -> str | None:
    """The configured WebSocket URL for *model*, or None if unknown/unset."""
    endpoint = _ENDPOINTS.get(model)
    if endpoint is None:
        return None
    url: str | None = getattr(settings, endpoint.url_attr, None)
    return url


def _warmup_pcm() -> bytes:
    """A quiet tone, not silence — silence can pass under VAD without inference."""
    n = int(_PCM_RATE * _WARMUP_SECONDS)
    amplitude = 3000
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * math.sin(2 * math.pi * 220 * i / _PCM_RATE)) for i in range(n)),
    )


class BasetenSTTProvider(WebSocketSTTProvider):
    """Baseten streaming STT provider (Whisper Large V3, Qwen3-ASR)."""

    _VALID_MODELS = frozenset(_ENDPOINTS)
    _SAMPLE_RATE = _PCM_RATE
    _NO_FINAL_ERROR = _NO_FINAL_ERROR

    def __init__(
        self,
        settings: Settings,
        model: str = "whisper-large-v3",
        open_timeout_s: float = _OPEN_TIMEOUT_S,
    ) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Baseten STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        super().__init__(settings, model)
        self._api_key = self._require_secret("baseten_api_key")
        endpoint = _ENDPOINTS[model]
        url = endpoint_url(settings, model)
        if not url:
            raise ValueError(f"{endpoint.url_attr} is required for the Baseten STT provider")
        self._url = url
        self._protocol = endpoint.protocol
        self._open_timeout_s = open_timeout_s

    @property
    def name(self) -> str:
        return "baseten"

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Wake and warm every configured endpoint concurrently; never fatal."""
        api_key = settings.baseten_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            return
        pcm = _warmup_pcm()

        async def _warm(model: str) -> None:
            provider = cls(settings, model, open_timeout_s=_WARMUP_TIMEOUT_S)
            t0 = time.monotonic()
            error: str | None = None
            try:
                async with asyncio.timeout(_WARMUP_TIMEOUT_S):
                    result = await provider.measure_ttft(pcm, 1, 2, _PCM_RATE)
                error = None if result.error == _NO_FINAL_ERROR else result.error
            except TimeoutError:
                error = f"warmup timed out after {_WARMUP_TIMEOUT_S}s; endpoint still not up"
            provider._log.info(
                "baseten_stt_prewarm",
                warmup_ms=round((time.monotonic() - t0) * 1000, 1),
                error=error,
            )

        targets = [model for model in _ENDPOINTS if endpoint_url(settings, model) is not None]
        if targets:
            await asyncio.gather(*(_warm(model) for model in targets), return_exceptions=True)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Api-Key {self._api_key.get_secret_value()}"}

    async def _connect(self, sample_rate: int) -> AbstractAsyncContextManager[Any]:
        return ws_client.connect(
            self._url,
            additional_headers=self._headers(),
            max_size=_MAX_WS_SIZE,
            open_timeout=self._open_timeout_s,
        )

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            json.dumps(
                {
                    "streaming_params": {
                        "enable_partial_transcripts": True,
                        "partial_transcript_interval_s": _PARTIAL_INTERVAL_S,
                    }
                }
            )
        )

    # Wholesale: the frame size is fixed by the protocol, not realtime_resolution.
    def _chunk_size(self, byte_rate: int, realtime_resolution: float) -> int:
        # Frame size is fixed by the protocol, not by the caller's resolution.
        return _FRAME_BYTES if self._protocol is _Protocol.BINARY else _APPEND_BYTES

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        if self._protocol is _Protocol.BINARY:
            if len(chunk) < _FRAME_BYTES:  # pad the final short frame to 512 samples
                chunk += b"\x00" * (_FRAME_BYTES - len(chunk))
            await ws.send(chunk)
            return
        await ws.send(
            json.dumps(
                {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()}
            )
        )

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # Signal end of audio so the server flushes the final transcript.
        terminator = (
            "end_audio" if self._protocol is _Protocol.BINARY else "input_audio_buffer.commit"
        )
        await ws.send(json.dumps({"type": terminator}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type = msg.get("type")
        if msg_type == "end_audio":
            # Sent twice: "acknowledged" (pre-transcripts) then "finished"
            # (true end). Stop only on "finished"; the session never closes.
            return bool(msg.get("body", {}).get("status") == "finished")
        if msg_type == "error":
            tx.fail(str(msg.get("message") or msg.get("body") or msg))
            self._log.warning("baseten_stt_error", msg=msg)
            return True
        if msg_type != "transcription":
            return False

        text = " ".join(str(seg.get("text", "")) for seg in msg.get("segments", [])).strip()
        if text:
            if msg.get("is_final"):
                tx.final(text, now)
            else:
                tx.partial(text, now)

        # The buffer protocol's end-of-stream marker; it sends no end_audio.
        return bool(msg.get("is_end_of_audio_flush"))
