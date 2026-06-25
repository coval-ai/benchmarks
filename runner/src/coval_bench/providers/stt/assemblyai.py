# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""AssemblyAI real-time STT provider (v3 streaming API).

Models: universal-streaming, universal-3.5-pro
Wire protocol: WebSocket, wss://streaming.assemblyai.com/v3/ws
Auth: Authorization: <key>
Close: {"type": "Terminate"}
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time  # monotonic clock — wall-clock can step on NTP sync
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

# Maps user-facing model names to the speech_model value required by the API.
# format_turns is omitted (default=false) — the formatting pass adds latency that
# would inflate TTFT measurements.
_SPEECH_MODEL_MAP: dict[str, str] = {
    "universal-streaming": "universal-streaming-english",
    "universal-3.5-pro": "universal-3-5-pro",
}
_WS_BASE = "wss://streaming.assemblyai.com/v3/ws"

# 1.0 = pure VAD silence-latency mode: the model never declares end-of-turn on a
# semantic guess, only on our ForceEndpoint at speech-end (TTFS parity). A lower
# value could let a confident short utterance auto-finalize before our signal.
_END_OF_TURN_CONFIDENCE_THRESHOLD = 1.0

# After ForceEndpoint, wait this long for the forced final before sending Terminate,
# so the close never races the final (the WS close-gate bug class). Falls through on
# timeout — the outer per-item timeout still bounds the run.
_FINAL_WAIT_S = 5.0


class AssemblyAIProvider(STTProvider):
    """AssemblyAI v3 streaming STT provider."""

    _VALID_MODELS = frozenset(_SPEECH_MODEL_MAP)

    def __init__(self, api_key: SecretStr, model: str = "universal-streaming") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid AssemblyAI model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"assemblyai-{self._model}"

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
        if sample_rate != 16000:
            result.error = f"AssemblyAI requires 16 kHz PCM input; got {sample_rate} Hz"
            return result

        total_start = time.monotonic()

        try:
            speech_model = _SPEECH_MODEL_MAP[self._model]
            url = (
                f"{_WS_BASE}?sample_rate={sample_rate}&speech_model={speech_model}"
                f"&end_of_turn_confidence_threshold={_END_OF_TURN_CONFIDENCE_THRESHOLD}"
            )
            headers = {"Authorization": self._api_key.get_secret_value()}
            final_event = asyncio.Event()
            async with ws_client.connect(url, additional_headers=headers) as ws:
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
                "assemblyai_measure_ttft_failed",
                provider="assemblyai",
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
        final_event: asyncio.Event,
    ) -> None:
        byte_rate = sample_width * sample_rate * channels
        data = audio_data
        first_chunk = True
        try:
            while data:
                chunk_size = int(byte_rate * realtime_resolution)
                chunk, data = data[:chunk_size], data[chunk_size:]
                if first_chunk:
                    result.audio_start_time = time.monotonic()
                    first_chunk = False
                await ws.send(chunk)
                await asyncio.sleep(realtime_resolution)
            # Force finalization at speech-end (TTFS parity), then wait for the final
            # before terminating so the close can't race it. Clear first: the event may
            # already be set by an earlier final, which would make the wait a no-op.
            final_event.clear()
            await ws.send(json.dumps({"type": "ForceEndpoint"}))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(final_event.wait(), timeout=_FINAL_WAIT_S)
            await ws.send(json.dumps({"type": "Terminate"}))
        except Exception as exc:
            logger.warning(
                "assemblyai_send_error", provider="assemblyai", model=self._model, exc_info=exc
            )
            raise

    async def _receive(
        self, ws: Any, result: TranscriptionResult, final_event: asyncio.Event
    ) -> None:
        complete_turns: list[str] = []
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type in ("Begin", "Termination"):
                    continue

                transcript = self._extract_transcript(msg)
                if transcript:
                    # TTFT fires on the first Turn regardless of end_of_turn. We want
                    # time-to-first-word, not time-to-first-completed-sentence.
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = now - result.audio_start_time
                        result.first_token_content = (
                            transcript[:30] + "..." if len(transcript) > 30 else transcript
                        )
                    result.partial_transcripts.append(transcript)

                if msg_type == "Turn" and msg.get("end_of_turn") and transcript:
                    complete_turns.append(transcript)
                    last_final_time = now
                    final_event.set()

        except Exception as exc:
            logger.warning(
                "assemblyai_receive_error",
                provider="assemblyai",
                model=self._model,
                exc_info=exc,
            )
            if result.error is None and last_final_time is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        if complete_turns:
            result.complete_transcript = " ".join(complete_turns).strip() or None
        elif result.partial_transcripts:
            result.complete_transcript = result.partial_transcripts[-1].strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())

    def _extract_transcript(self, msg: dict[str, Any]) -> str:
        if msg.get("type") != "Turn":
            return ""
        transcript = str(msg.get("transcript", "")).strip()
        if transcript:
            return transcript
        words: list[dict[str, Any]] = msg.get("words", [])
        parts = [
            str(w["text"]).strip()
            for w in words
            if isinstance(w, dict) and str(w.get("text", "")).strip()
        ]
        return " ".join(parts)
