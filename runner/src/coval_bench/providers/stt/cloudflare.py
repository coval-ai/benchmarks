# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloudflare Workers AI real-time STT provider.

Supports the Deepgram nova models Workers AI hosts (nova-3). The socket relays
Deepgram's v1/listen protocol unchanged:
  wss://api.cloudflare.com/client/v4/accounts/<account>/ai/run/@cf/deepgram/<model>
Auth: Authorization: Bearer <api token>
Finalize: {"type": "Finalize"}   Close: {"type": "CloseStream"}
"""

from __future__ import annotations

import asyncio
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

_WS_BASE = "wss://api.cloudflare.com/client/v4/accounts"

# After Finalize, wait this long for the forced final before sending CloseStream,
# so the close can't race the final. Falls through on timeout; the outer per-item
# timeout still bounds the run.
_FINAL_WAIT_S = 5.0


class CloudflareSTTProvider(STTProvider):
    """Deepgram nova STT served from Cloudflare Workers AI."""

    def __init__(self, api_key: SecretStr | None, model: str, account_id: str | None) -> None:
        if api_key is None or not api_key.get_secret_value():
            raise ValueError("cloudflare_api_key is required in Settings")
        if not account_id:
            raise ValueError("cloudflare_account_id is required in Settings")
        self._api_key = api_key
        self._model = model
        self._account_id = account_id

    @property
    def name(self) -> str:
        return f"cloudflare-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int, channels: int) -> str:
        return (
            f"{_WS_BASE}/{self._account_id}/ai/run/@cf/deepgram/{self._model}"
            f"?sample_rate={sample_rate}"
            f"&encoding=linear16"
            f"&channels={channels}"
            f"&interim_results=true"
            f"&vad_events=true"
            f"&no_delay=true"
            f"&punctuate=true"
            f"&filler_words=true"
            # Native silence endpointing off; the final is forced via Finalize at
            # speech-end instead (TTFS parity with the direct Deepgram provider).
            f"&endpointing=false"
        )

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        total_start = time.monotonic()

        try:
            url = self._build_websocket_url(sample_rate, channels)
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

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
                "cloudflare_measure_ttft_failed",
                provider="cloudflare",
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
        chunk_size = int(byte_rate * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, byte_rate):
                result.audio_start_time = start
                await ws.send(chunk)
            # Clear first: nova can emit an is_final segment mid-stream, which would
            # leave the latch set and make the wait a no-op.
            final_event.clear()
            await ws.send(json.dumps({"type": "Finalize"}))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(final_event.wait(), timeout=_FINAL_WAIT_S)
            await ws.send(json.dumps({"type": "CloseStream"}))
        except Exception as exc:
            logger.warning(
                "cloudflare_send_error", provider="cloudflare", model=self._model, exc_info=exc
            )
            raise

    async def _receive(
        self, ws: Any, result: TranscriptionResult, final_event: asyncio.Event
    ) -> None:
        final_segments: list[str] = []
        pending_partial: str = ""
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "SpeechStarted":
                    if result.audio_start_time is not None:
                        elapsed = now - result.audio_start_time
                        if result.vad_first_detected is None:
                            result.vad_first_detected = elapsed
                            result.vad_first_event_content = str(msg)
                    result.vad_events_count = (result.vad_events_count or 0) + 1
                    continue

                if msg_type != "Results":
                    continue

                transcript = _extract_transcript(msg)
                if not transcript:
                    continue

                if result.ttft_seconds is None and result.audio_start_time is not None:
                    result.ttft_seconds = now - result.audio_start_time
                    result.first_token_content = (
                        transcript[:30] + "..." if len(transcript) > 30 else transcript
                    )

                result.partial_transcripts.append(transcript)

                # Full transcript is the concatenation of ALL is_final segments;
                # speech_final alone drops the is_final-only pieces between pauses.
                if msg.get("is_final"):
                    final_segments.append(transcript)
                    pending_partial = ""
                    last_final_time = now
                    final_event.set()
                elif len(transcript) > len(pending_partial):
                    pending_partial = transcript

        except Exception as exc:
            logger.warning(
                "cloudflare_receive_error", provider="cloudflare", model=self._model, exc_info=exc
            )
            if result.error is None and last_final_time is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        if final_segments or pending_partial:
            parts = list(final_segments)
            if pending_partial:
                parts.append(pending_partial)
            result.complete_transcript = " ".join(parts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())


def _extract_transcript(msg: dict[str, Any]) -> str:
    # {"type": "Results", "channel": {"alternatives": [{"transcript": "...", "words": [...]}]}}
    try:
        alternatives: list[dict[str, Any]] = msg.get("channel", {}).get("alternatives", [])
        if not alternatives:
            return ""
        alt = alternatives[0]
        parts = [
            text
            for w in alt.get("words", [])
            if (text := str(w.get("punctuated_word", "")).strip() or str(w.get("word", "")).strip())
        ]
        if parts:
            return " ".join(parts)
        return str(alt.get("transcript", "")).strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
