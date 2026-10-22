# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Realtime API STT provider.

Benchmarks the realtime transcription models — ``gpt-realtime-whisper``,
``gpt-4o-transcribe``, and ``gpt-4o-mini-transcribe`` — over a single websocket
transcription session.

Every model runs single-pass manual-commit: audio streams in, then an explicit
``input_audio_buffer.commit`` finalizes the buffer, so AudioToFinal carries an extra
post-audio round-trip versus auto-finalizing streaming providers.

Streaming differs by model. ``gpt-realtime-whisper`` emits partial deltas in real time, so
its TTFT is comparable to streaming ASR; it also rejects ``turn_detection: server_vad``
outright, leaving manual-commit the only option (verified 2026-06). ``gpt-4o-transcribe``
and ``gpt-4o-mini-transcribe`` only transcribe a finalized segment — no partials
mid-utterance, even with server VAD — so their TTFT is excluded upstream (see
``METRIC_EXCLUSIONS`` in the metrics registry).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy import signal

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._realtime import RealtimeSTTProvider
from coval_bench.providers.stt._streaming import Transcription

_WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
_INPUT_SAMPLE_RATE = 24000
_READY_TIMEOUT_S = 30.0

signal.resample_poly(np.zeros(8, dtype=np.float32), up=3, down=2)


@dataclass
class _OpenAITranscription(Transcription):
    # Deltas are increments scoped to an item; the running hypothesis is their concatenation.
    partial_by_item: defaultdict[str, str] = field(default_factory=lambda: defaultdict(str))


class OpenAISTTProvider(RealtimeSTTProvider):
    """OpenAI Realtime API STT provider for the realtime transcription models."""

    _LABEL = "OpenAI"

    def __init__(self, settings: Settings, model: str = "gpt-realtime-whisper") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("openai_api_key")

    @property
    def name(self) -> str:
        return f"openai-{self._model}"

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _OpenAITranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        return _WS_URL

    def _session(self) -> dict[str, Any]:
        return {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": _INPUT_SAMPLE_RATE},
                    "transcription": {"model": self._model, "language": "en"},
                    # server_vad is rejected for this model; null = manual-commit only.
                    "turn_detection": None,
                }
            },
        }

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(self._frame("session.update", session=self._session()))
        await self._await_ready(ws, "session.updated", _READY_TIMEOUT_S)

    def _prepare_audio(self, audio_data: bytes, sample_rate: int) -> tuple[bytes, int]:
        byte_rate = _INPUT_SAMPLE_RATE * 2
        if sample_rate == _INPUT_SAMPLE_RATE:
            return audio_data, byte_rate
        pcm = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        gcd = math.gcd(sample_rate, _INPUT_SAMPLE_RATE)
        resampled = signal.resample_poly(pcm, up=_INPUT_SAMPLE_RATE // gcd, down=sample_rate // gcd)
        clipped = np.clip(np.round(resampled * 32767.0), -32768, 32767).astype(np.int16)
        return cast(bytes, clipped.tobytes()), byte_rate

    def _error_message(self, event: dict[str, Any]) -> str:
        return str(event.get("error", {}).get("message", "OpenAI error"))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_OpenAITranscription, tx)
        event_type = str(msg.get("type", ""))

        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = str(msg.get("delta", ""))
            if delta.strip():
                item_id = str(msg.get("item_id", "default"))
                tx.partial_by_item[item_id] += delta
                tx.partial(tx.partial_by_item[item_id].strip(), now)
            return False

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(msg.get("transcript", "")).strip()
            if not transcript:
                return False
            tx.partial(transcript, now)
            tx.final(transcript, now)
            return True

        if event_type == "conversation.item.input_audio_transcription.failed":
            tx.fail(str(msg.get("error", {}).get("message", "OpenAI transcription failed")))
            self._log.warning("openai_stt_error", msg=msg)
            return True

        if event_type == "error":
            tx.fail(self._error_message(msg))
            self._log.warning("openai_stt_error", msg=msg)
            return True

        return False
