# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Together AI real-time STT provider (WebSocket).

Wire protocol (OpenAI-Realtime style):
  URL:   wss://api.together.ai/v1/realtime?model=<id>&input_audio_format=pcm_s16le_16000
  Auth:  Authorization: Bearer <key> + OpenAI-Beta: realtime=v1
  Send:  {"type": "input_audio_buffer.append", "audio": <base64 PCM>} then
         {"type": "input_audio_buffer.commit"} to force the final.
  Recv:  conversation.item.input_audio_transcription.delta (partial) /
         .completed (final). The server never closes; the client closes after
         draining the forced final.

The Nemotron models run with ~1.1 s of encoder lookahead and commit drops it,
truncating the tail. Trailing silence is paced in before the commit to flush
the lookahead (Flux/Rev AI pattern); their TTFS is excluded in
``registries/metrics.py`` since the final's timing then tracks the client's
silence length, not the engine.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks

logger = structlog.get_logger(__name__)

_MODEL_IDS = {
    "nemotron-3-asr-streaming-0.6b": "nvidia/nemotron-3-asr-streaming-0.6b",
    "nemotron-3.5-asr-streaming-0.6b": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "whisper-large-v3": "openai/whisper-large-v3",
}

# Models whose commit drops the encoder-lookahead buffer; they get trailing
# silence before the commit so the tail is decoded.
_LOOKAHEAD_FLUSH_MODELS = frozenset(
    {"nemotron-3-asr-streaming-0.6b", "nemotron-3.5-asr-streaming-0.6b"}
)

# Nemotron's deepest lookahead config is 1.12 s; pad past it.
_TAIL_SILENCE_S = 1.6

# After commit, wait this long for the forced final before closing, so the
# close can't race the final.
_FINAL_WAIT_S = 5.0


class TogetherSTTProvider(STTProvider):
    """Together AI streaming STT provider."""

    _VALID_MODELS = frozenset(_MODEL_IDS)

    def __init__(self, api_key: SecretStr | None, model: str = "whisper-large-v3") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Together STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None:
            raise ValueError("together_api_key is required for the Together STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"together-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)
        if sample_rate != 16000 or channels != 1 or sample_width != 2:
            result.error = (
                "Together requires 16 kHz mono 16-bit PCM input; got "
                f"sample_rate={sample_rate}, channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            url = (
                "wss://api.together.ai/v1/realtime"
                f"?model={_MODEL_IDS[self._model]}"
                "&input_audio_format=pcm_s16le_16000"
            )
            headers = {
                "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                "OpenAI-Beta": "realtime=v1",
            }
            final_event = asyncio.Event()
            async with ws_client.connect(url, additional_headers=headers) as ws:
                send_task = asyncio.create_task(
                    self._send_audio(
                        ws, audio_data, sample_rate, result, realtime_resolution, final_event
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
                "together_measure_ttft_failed",
                provider="together",
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
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
        final_event: asyncio.Event,
    ) -> None:
        byte_rate = sample_rate * 2
        chunk_size = int(byte_rate * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, byte_rate):
                result.audio_start_time = start
                await self._append(ws, chunk)
            if self._model in _LOOKAHEAD_FLUSH_MODELS:
                silence = bytes(int(byte_rate * _TAIL_SILENCE_S))
                async for chunk, _ in paced_chunks(silence, chunk_size, byte_rate):
                    await self._append(ws, chunk)
            # Clear first: a mid-stream turn final would leave the latch set
            # and make the wait a no-op.
            final_event.clear()
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(final_event.wait(), timeout=_FINAL_WAIT_S)
            # The server holds the socket open; close so the receive loop ends.
            await ws.close()
        except Exception as exc:
            logger.warning(
                "together_send_error", provider="together", model=self._model, exc_info=exc
            )
            raise

    @staticmethod
    async def _append(ws: Any, chunk: bytes) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )

    async def _receive(
        self, ws: Any, result: TranscriptionResult, final_event: asyncio.Event
    ) -> None:
        final_segments: list[str] = []
        # Deltas are cumulative running transcripts revised in place, keyed by
        # conversation item — keep only each item's latest.
        item_deltas: dict[str, str] = {}
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "error":
                    error_obj = msg.get("error")
                    detail = (
                        error_obj.get("message") if isinstance(error_obj, dict) else None
                    ) or msg.get("message")
                    result.error = str(detail or msg)
                    logger.warning(
                        "together_stt_error", provider="together", model=self._model, msg=msg
                    )
                    final_event.set()
                    break

                if msg_type.endswith("input_audio_transcription.delta"):
                    delta = str(msg.get("delta", ""))
                    if not delta.strip():
                        continue
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = now - result.audio_start_time
                        snippet = delta.strip()
                        result.first_token_content = (
                            snippet[:30] + "..." if len(snippet) > 30 else snippet
                        )
                    result.partial_transcripts.append(delta.strip())
                    item_deltas[str(msg.get("item_id", ""))] = delta
                    continue

                if msg_type.endswith("input_audio_transcription.completed"):
                    transcript = str(msg.get("transcript", "")).strip()
                    if transcript:
                        final_segments.append(transcript)
                        last_final_time = now
                    final_event.set()

        except Exception as exc:
            logger.warning(
                "together_receive_error", provider="together", model=self._model, exc_info=exc
            )
            if result.error is None and last_final_time is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        if final_segments:
            result.complete_transcript = " ".join(final_segments).strip() or None
        elif item_deltas:
            result.complete_transcript = (
                " ".join(d.strip() for d in item_deltas.values()).strip() or None
            )

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
