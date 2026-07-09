# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rev AI real-time streaming STT provider (Reverb model).

Wire protocol: WebSocket, wss://api.rev.ai/speechtotext/v1/stream
Auth: ``access_token`` query parameter.
Audio in: raw binary PCM frames.
Close: an ``"EOS"`` text frame; the server flushes the last final and closes.
Server messages (serialised JSON, keyed by ``type``):
  connected -> handshake ack, no transcript
  partial   -> best-guess hypothesis, ``elements: [{type: "text", value}]``
  final     -> committed segment, ``elements: [{type: "text"|"punct", value, ...}]``

``transcriber=machine_v2`` pins the Reverb ASR model rather than whatever
default is in effect for the account (see docs.rev.ai/api/streaming).
"""

from __future__ import annotations

import asyncio
import json
import time  # monotonic clock — wall-clock can step on NTP sync
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

_WS_BASE = "wss://api.rev.ai/speechtotext/v1/stream"
_TRANSCRIBER_FOR_MODEL: dict[str, str] = {"reverb": "machine_v2"}
_EOS = "EOS"

# Rev has no force-finalize: EOS discards the un-endpointed tail, so we feed
# trailing silence to make Reverb endpoint the last segment (cap; breaks early).
_EOT_SILENCE_S = 2.0


class RevAISTTProvider(STTProvider):
    """Rev AI streaming STT provider."""

    _VALID_MODELS = frozenset(_TRANSCRIBER_FOR_MODEL)

    def __init__(self, api_key: SecretStr | None, model: str = "reverb") -> None:
        if not self._model_supported(model):
            raise ValueError(f"Invalid Rev AI model {model!r}. Valid: {sorted(self._VALID_MODELS)}")
        if api_key is None:
            raise ValueError("revai_api_key is required for the Rev AI STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "revai"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int) -> str:
        params = {
            "access_token": self._api_key.get_secret_value(),
            "content_type": (
                f"audio/x-raw;layout=interleaved;rate={sample_rate};format=S16LE;channels=1"
            ),
            "language": "en",
            "transcriber": _TRANSCRIBER_FOR_MODEL[self._model],
        }
        return f"{_WS_BASE}?{urlencode(params)}"

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
                "Rev AI requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            url = self._build_websocket_url(sample_rate)
            final_event = asyncio.Event()
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
                        final_event,
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result, final_event))
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
                "revai_measure_ttft_failed", provider="revai", model=self._model, exc_info=exc
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
        final_event: asyncio.Event,
    ) -> None:
        byte_rate = sample_width * sample_rate * channels
        chunk_size = int(byte_rate * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, byte_rate):
                result.audio_start_time = start
                await ws.send(chunk)
            # Clear first: a mid-stream final would leave the latch set and skip the
            # drain. Silence never sets audio_start_time (audio_to_final is from speech).
            final_event.clear()
            silence = bytes(int(byte_rate * _EOT_SILENCE_S))
            async for chunk, _ in paced_chunks(silence, chunk_size, byte_rate):
                if final_event.is_set():
                    break
                await ws.send(chunk)
            await ws.send(_EOS)
        except Exception as exc:
            logger.warning("revai_send_error", provider="revai", model=self._model, exc_info=exc)
            raise

    async def _receive(
        self, ws: Any, result: TranscriptionResult, final_event: asyncio.Event
    ) -> None:
        final_segments: list[str] = []
        # A partial left standing at close is the un-finalized tail; recover it.
        pending_partial: str = ""
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "connected":
                    continue

                transcript = self._elements_to_text(msg.get("elements", []))
                if not transcript:
                    continue

                # TTFT is time-to-first-word: fire on the first partial, not the first final.
                set_first_token(result, transcript, now=now)
                add_partial_transcript(result, transcript)

                if msg_type == "final":
                    final_segments.append(transcript)
                    pending_partial = ""
                    last_final_time = now
                    final_event.set()
                elif len(transcript) > len(pending_partial):
                    pending_partial = transcript

        except Exception as exc:
            logger.warning("revai_receive_error", provider="revai", model=self._model, exc_info=exc)
            if result.error is None and last_final_time is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        segments = final_segments + ([pending_partial] if pending_partial else [])
        finalize_transcript(result, final_segments=segments)

    @staticmethod
    def _elements_to_text(elements: list[dict[str, Any]]) -> str:
        """Join the spoken-word (``text``) elements; ``punct`` is dropped for WER parity."""
        parts = [
            str(el.get("value", "")).strip()
            for el in elements
            if isinstance(el, dict)
            and el.get("type") == "text"
            and str(el.get("value", "")).strip()
        ]
        return " ".join(parts)
