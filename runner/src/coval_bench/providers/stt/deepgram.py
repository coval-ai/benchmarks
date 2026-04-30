# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepgram real-time STT provider.

Supports models: default (nova-2), nova-2, nova-3, flux-general-en.
Wire protocol: WebSocket, wss://api.deepgram.com/v1/listen
Auth: Authorization: Token <key>
Close: {"type": "CloseStream"}
"""

from __future__ import annotations

import asyncio
import json
import time  # monotonic clock — wall-clock can step on NTP sync
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

_VALID_MODELS = ("default", "nova-2", "nova-3", "flux-general-en")


class DeepgramProvider(STTProvider):
    """Deepgram streaming STT provider."""

    def __init__(self, api_key: SecretStr, model: str = "default") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Deepgram model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        if self._model == "default":
            return "deepgram"
        return f"deepgram-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # URL building
    # ------------------------------------------------------------------

    def _build_websocket_url(self, sample_rate: int, channels: int) -> str:
        if self._model == "flux-general-en":
            # Without interim_results+no_delay the v2/listen preview endpoint
            # buffers transcripts until CloseStream, leaving ttft unmeasurable.
            return (
                "wss://api.preview.deepgram.com/v2/listen"
                "?model=flux-general-en&sample_rate=16000&encoding=linear16"
                "&interim_results=true&no_delay=true"
            )
        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?sample_rate={sample_rate}"
            f"&encoding=linear16"
            f"&channels={channels}"
            f"&interim_results=true"
            f"&vad_events=true"
            f"&no_delay=true"
        )
        if self._model != "default":
            url += f"&model={self._model}"
        return url

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        total_start = time.monotonic()

        try:
            url = self._build_websocket_url(sample_rate, channels)
            headers = {"Authorization": f"Token {self._api_key.get_secret_value()}"}

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
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

        except Exception as exc:
            logger.exception("deepgram measure_ttft failed", error=str(exc))
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

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
            await ws.send(json.dumps({"type": "CloseStream"}))
        except Exception as exc:
            logger.exception("deepgram send error", error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Receive helpers
    # ------------------------------------------------------------------

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_segments: list[str] = []
        flux_latest: str = ""
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                # VAD events
                if msg_type == "SpeechStarted":
                    if result.audio_start_time is not None:
                        elapsed = now - result.audio_start_time
                        if result.vad_first_detected is None:
                            result.vad_first_detected = elapsed
                            result.vad_first_event_content = str(msg)
                    result.vad_events_count = (result.vad_events_count or 0) + 1
                    continue

                if msg_type in ("SpeechEnded", "Metadata", "Connected", "TurnInfo"):
                    continue

                transcript = self._extract_transcript(msg)
                if not transcript:
                    continue

                # TTFT
                if result.ttft_seconds is None and result.audio_start_time is not None:
                    result.ttft_seconds = now - result.audio_start_time
                    result.first_token_content = (
                        transcript[:30] + "..." if len(transcript) > 30 else transcript
                    )

                result.partial_transcripts.append(transcript)

                if self._model == "flux-general-en":
                    flux_latest = transcript
                    # flux emits rolling updates; every transcript update is the
                    # latest "final" view of what was said — track the last one.
                    last_final_time = now
                elif self._model in ("nova-2", "nova-3"):
                    if msg.get("speech_final"):
                        final_segments.append(transcript)
                        last_final_time = now
                else:
                    # default model — treat is_final as final
                    if msg.get("is_final"):
                        final_segments.append(transcript)
                        last_final_time = now

        except Exception as exc:
            logger.exception("deepgram receive error", error=str(exc))

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        # Build complete transcript
        if self._model == "flux-general-en":
            result.complete_transcript = flux_latest.strip() or None
        elif final_segments:
            result.complete_transcript = " ".join(final_segments).strip()
        elif result.partial_transcripts:
            result.complete_transcript = max(result.partial_transcripts, key=len).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())

    # ------------------------------------------------------------------
    # Transcript extraction
    # ------------------------------------------------------------------

    def _extract_transcript(self, msg: dict[str, Any]) -> str:
        # Both the standard endpoint (v1/listen) and the flux preview endpoint
        # (v2/listen) return Deepgram's standard Results message shape:
        #   {"type": "Results", "channel": {"alternatives": [{"transcript": "..."}]}}
        # An earlier implementation read a top-level "transcript" key for flux,
        # but that key does not exist in a Results message — causing silent NULL
        # TTFT for flux-general-en on every run.
        try:
            channel: dict[str, Any] = msg.get("channel", {})
            alternatives: list[dict[str, Any]] = channel.get("alternatives", [])
            if not alternatives:
                return ""
            alt = alternatives[0]
            words: list[dict[str, Any]] = alt.get("words", [])
            if words:
                parts = []
                for w in words:
                    text = (
                        str(w.get("punctuated_word", "")).strip() or str(w.get("word", "")).strip()
                    )
                    if text:
                        parts.append(text)
                if parts:
                    return " ".join(parts)
            return str(alt.get("transcript", "")).strip()
        except (KeyError, IndexError, TypeError):
            return ""
