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
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks

if TYPE_CHECKING:
    from coval_bench.config import Settings

logger = structlog.get_logger(__name__)

# Silero VAD frame: 512 samples * 2 bytes = 1024 bytes; other sizes are rejected.
_FRAME_SAMPLES = 512
_FRAME_BYTES = _FRAME_SAMPLES * 2
_SAMPLE_RATE = 16000
_BYTE_RATE = _SAMPLE_RATE * 2  # 16 kHz mono 16-bit
_APPEND_BYTES = int(_SAMPLE_RATE * 0.1) * 2
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
    n = int(_SAMPLE_RATE * _WARMUP_SECONDS)
    amplitude = 3000
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * math.sin(2 * math.pi * 220 * i / _SAMPLE_RATE)) for i in range(n)),
    )


class BasetenSTTProvider(STTProvider):
    """Baseten streaming STT provider (Whisper Large V3, Qwen3-ASR)."""

    _VALID_MODELS = frozenset(_ENDPOINTS)

    def __init__(
        self,
        api_key: SecretStr | None,
        model: str = "whisper-large-v3",
        ws_url: str | None = None,
        open_timeout_s: float = _OPEN_TIMEOUT_S,
    ) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Baseten STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None or not api_key.get_secret_value().strip():
            raise ValueError("baseten_api_key is required for the Baseten STT provider")
        endpoint = _ENDPOINTS[model]
        if not ws_url:
            raise ValueError(f"{endpoint.url_attr} is required for the Baseten STT provider")
        self._api_key = api_key
        self._model = model
        self._ws_url = ws_url
        self._protocol = endpoint.protocol
        self._open_timeout_s = open_timeout_s

    @property
    def name(self) -> str:
        return "baseten"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Wake and warm every configured endpoint concurrently; never fatal."""
        api_key = settings.baseten_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            return
        pcm = _warmup_pcm()

        async def _warm(model: str, url: str) -> None:
            provider = cls(
                api_key=api_key, model=model, ws_url=url, open_timeout_s=_WARMUP_TIMEOUT_S
            )
            t0 = time.monotonic()
            error: str | None = None
            try:
                async with asyncio.timeout(_WARMUP_TIMEOUT_S):
                    result = await provider.measure_ttft(pcm, 1, 2, _SAMPLE_RATE)
                error = None if result.error == _NO_FINAL_ERROR else result.error
            except TimeoutError:
                error = f"warmup timed out after {_WARMUP_TIMEOUT_S}s; endpoint still not up"
            logger.info(
                "baseten_stt_prewarm",
                provider="baseten",
                model=model,
                warmup_ms=round((time.monotonic() - t0) * 1000, 1),
                error=error,
            )

        targets = [
            (model, url)
            for model in _ENDPOINTS
            if (url := endpoint_url(settings, model)) is not None
        ]
        if targets:
            await asyncio.gather(
                *(_warm(model, url) for model, url in targets), return_exceptions=True
            )

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)
        if sample_rate != _SAMPLE_RATE:
            result.error = f"Baseten requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Baseten requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()
        headers = {"Authorization": f"Api-Key {self._api_key.get_secret_value()}"}

        try:
            async with ws_client.connect(
                self._ws_url,
                additional_headers=headers,
                max_size=_MAX_WS_SIZE,
                open_timeout=self._open_timeout_s,
            ) as ws:
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

                send_task = asyncio.create_task(self._send_audio(ws, audio_data, result))
                recv_task = asyncio.create_task(self._receive(ws, result))
                tasks = (send_task, recv_task)
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                if any(not task.cancelled() and task.exception() is not None for task in done):
                    for task in pending:
                        task.cancel()
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                if result.error is None and result.audio_to_final_seconds is None:
                    for outcome in outcomes:
                        if isinstance(outcome, Exception):
                            result.error = str(outcome)
                            break
                    else:
                        result.error = _NO_FINAL_ERROR

        except Exception as exc:
            logger.warning(
                "baseten_measure_ttft_failed", provider="baseten", model=self._model, exc_info=exc
            )
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        result: TranscriptionResult,
    ) -> None:
        binary = self._protocol is _Protocol.BINARY
        chunk_bytes = _FRAME_BYTES if binary else _APPEND_BYTES
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_bytes, _BYTE_RATE):
                # Stop early if _receive already recorded a protocol/auth error.
                if result.error is not None:
                    break
                result.audio_start_time = start
                if binary:
                    frame = chunk
                    if len(frame) < _FRAME_BYTES:  # pad the final short frame to 512 samples
                        frame += b"\x00" * (_FRAME_BYTES - len(frame))
                    await ws.send(frame)
                else:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode(),
                            }
                        )
                    )
            # Signal end of audio so the server flushes the final transcript.
            terminator = "end_audio" if binary else "input_audio_buffer.commit"
            await ws.send(json.dumps({"type": terminator}))
        except Exception as exc:
            logger.warning(
                "baseten_send_error", provider="baseten", model=self._model, exc_info=exc
            )
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        # Per-segment finals accumulate in arrival order (Whisper + Silero VAD).
        final_parts: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue  # STT replies are JSON; ignore any binary frames

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()

                msg_type = msg.get("type")
                if msg_type == "end_audio":
                    # Sent twice: "acknowledged" (pre-transcripts) then "finished"
                    # (true end). Stop only on "finished"; the session never closes.
                    if msg.get("body", {}).get("status") == "finished":
                        break
                    continue
                if msg_type == "error":
                    result.error = str(msg.get("message") or msg.get("body") or msg)
                    logger.warning(
                        "baseten_stt_error", provider="baseten", model=self._model, msg=msg
                    )
                    break
                if msg_type != "transcription":
                    continue

                text = " ".join(str(seg.get("text", "")) for seg in msg.get("segments", [])).strip()
                if text:
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = now - result.audio_start_time
                        result.first_token_content = text[:30] + "..." if len(text) > 30 else text

                    if msg.get("is_final"):
                        final_parts.append(text)
                        if result.audio_start_time is not None:
                            result.audio_to_final_seconds = now - result.audio_start_time
                    else:
                        result.partial_transcripts.append(text)

                # The buffer protocol's end-of-stream marker; it sends no end_audio.
                if msg.get("is_end_of_audio_flush"):
                    break

        except Exception as exc:
            logger.warning(
                "baseten_receive_error", provider="baseten", model=self._model, exc_info=exc
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        if final_parts:
            result.complete_transcript = " ".join(final_parts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
