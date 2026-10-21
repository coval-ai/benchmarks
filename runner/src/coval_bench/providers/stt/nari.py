# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Nari Labs realtime STT provider (WebSocket).

Nari serves Qwen3-ASR behind an OpenAI-Realtime-shaped transcription socket.
The session runs manual-commit (``turn_detection: null``): audio streams in as
base64 PCM16 appends, then ``input_audio_buffer.commit`` finalizes the buffer.
Recordings longer than the session's ``max_duration_seconds`` (36 s) are split
server-side into several items, each finalized with ``commit_reason:
"max_duration"``; the manual commit produces the last item.
Protocol: https://docs.narilabs.com/api-reference/speech-to-text/realtime/realtime-transcription
"""

from __future__ import annotations

from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.stt._realtime import RealtimeSTTProvider
from coval_bench.providers.stt._streaming import Transcription

_WS_URL = "wss://api.narilabs.com/v1/realtime?intent=transcription"
_READY_TIMEOUT_S = 30.0


class NariSTTProvider(RealtimeSTTProvider):
    """Nari Labs streaming STT provider for the hosted Qwen3-ASR models."""

    _LABEL = "Nari"
    _SAMPLE_RATE = 16000

    def __init__(self, settings: Settings, model: str = "qwen3-asr-fast") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("nari_api_key")

    @property
    def name(self) -> str:
        return "nari"

    def _ws_url(self, sample_rate: int) -> str:
        return _WS_URL

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            self._frame(
                "session.configure",
                session={"model": self._model, "language": "en", "turn_detection": None},
            )
        )
        await self._await_ready(ws, "session.configured", _READY_TIMEOUT_S)

    def _error_message(self, event: dict[str, Any]) -> str:
        error = event.get("error") or {}
        message = str(error.get("message") or "Nari error")
        request_id = error.get("requestId")
        return f"{message} (requestId={request_id})" if request_id else message

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        event_type = str(msg.get("type", ""))

        if event_type == "transcript.partial":
            tx.partial(str(msg.get("transcript", "")), now)
            return False

        if event_type == "transcript.completed":
            transcript = str(msg.get("transcript", "")).strip()
            tx.partial(transcript, now)
            if msg.get("commit_reason") == "max_duration":
                # An auto-split item is not the final; only the manual commit stamps it.
                if transcript:
                    tx.finals.append(transcript)
                return False
            tx.final(transcript, now)
            return True

        if event_type == "input_audio_buffer.commit_empty":
            tx.fail("Nari commit had no pending audio")
            self._log.warning("nari_stt_error", msg=msg)
            return True

        if event_type == "error":
            tx.fail(self._error_message(msg))
            self._log.warning("nari_stt_error", msg=msg)
            return True

        return False
