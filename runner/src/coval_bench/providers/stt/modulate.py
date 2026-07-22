# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Modulate (Velma-2) real-time streaming STT provider.

Wire protocol: WebSocket, one endpoint per model; the model id is the endpoint
path segment (``velma-2-stt-streaming`` is multilingual,
``velma-2-stt-streaming-english-v2`` is fast English-only).
Auth: ``api_key`` query parameter (no header).
Audio in: raw binary PCM frames; format declared via query params
  (``audio_format=s16le&sample_rate=<hz>&num_channels=1``).
Close: an empty ``""`` text frame flushes the buffer; the server returns the
  final ``utterance`` message(s), then a ``done`` message, then closes.
Server messages (serialised JSON, keyed by ``type``):
  partial_utterance -> in-progress preview, ``partial_utterance.text``
  utterance         -> committed segment, ``utterance.text``
  done              -> stream complete, ``duration_ms``
  error             -> ``error`` description

The multilingual endpoint needs ``partial_results=true`` to emit partials and
runs with ``speaker_diarization=false`` so a single utterance is not split into
per-speaker finals; the finals it does emit are concatenated in arrival order.

The English endpoint emits partials on a fixed ~1.5 s cadence, so its TTFT
tracks the emission interval rather than engine latency and is excluded in
``registries/metrics.py``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)

logger = structlog.get_logger(__name__)

_WS_BASE = "wss://platform.modulate.ai/api"
_EOS = ""

_MULTILINGUAL_MODEL = "velma-2-stt-streaming"
_ENGLISH_MODEL = "velma-2-stt-streaming-english-v2"


class ModulateSTTProvider(STTProvider):
    """Modulate streaming STT provider."""

    _VALID_MODELS = frozenset({_MULTILINGUAL_MODEL, _ENGLISH_MODEL})

    def __init__(self, api_key: SecretStr | None, model: str = _ENGLISH_MODEL) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Modulate model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None:
            raise ValueError("modulate_api_key is required for the Modulate STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"modulate-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int) -> str:
        params: dict[str, str] = {
            "api_key": self._api_key.get_secret_value(),
            "audio_format": "s16le",
            "sample_rate": str(sample_rate),
            "num_channels": "1",
        }
        if self._model == _MULTILINGUAL_MODEL:
            params["partial_results"] = "true"
            params["speaker_diarization"] = "false"
        return f"{_WS_BASE}/{self._model}?{urlencode(params)}"

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)
        if channels != 1 or sample_width != 2:
            result.error = (
                "Modulate requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            url = self._build_websocket_url(sample_rate)
            async with ws_client.connect(url) as ws:
                send_task = asyncio.create_task(
                    self._send_audio(
                        ws,
                        audio_data,
                        channels,
                        sample_width,
                        sample_rate,
                        result,
                        realtime_resolution,
                    )
                )
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

        except Exception as exc:
            logger.warning(
                "modulate_measure_ttft_failed",
                provider="modulate",
                model=self._model,
                exc_info=exc,
            )
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
    ) -> None:
        byte_rate = sample_width * sample_rate * channels
        chunk_size = int(byte_rate * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, byte_rate):
                result.audio_start_time = start
                await ws.send(chunk)
            await ws.send(_EOS)
        except Exception as exc:
            logger.warning(
                "modulate_send_error", provider="modulate", model=self._model, exc_info=exc
            )
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_segments: list[str] = []
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "error":
                    result.error = str(msg.get("error") or msg)
                    logger.warning(
                        "modulate_stt_error", provider="modulate", model=self._model, msg=msg
                    )
                    break

                if msg_type == "done":
                    break

                payload = msg.get(msg_type)
                transcript = (
                    str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
                )
                if not transcript:
                    continue

                set_first_token(result, transcript, now=now)
                add_partial_transcript(result, transcript)

                if msg_type == "utterance":
                    final_segments.append(transcript)
                    last_final_time = now

        except Exception as exc:
            logger.warning(
                "modulate_receive_error", provider="modulate", model=self._model, exc_info=exc
            )
            if result.error is None and last_final_time is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        finalize_transcript(result, final_segments=final_segments)
