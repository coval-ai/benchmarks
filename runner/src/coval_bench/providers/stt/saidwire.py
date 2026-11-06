# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Saidwire real-time STT provider.

Speaks the Deepgram v1/listen protocol (Results events, Finalize and CloseStream
control frames) on Saidwire's own host:
  wss://us-east.api.saidwire.com/v1/listen
Auth: Authorization: Token <key>
English only, one channel per stream. The model query param is accepted but every
stream runs the current saidwire-en build; the version served is in the Results
metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = "wss://us-east.api.saidwire.com/v1/listen"


@dataclass
class _SaidwireTranscription(Transcription):
    # Longest interim not yet covered by a final; appended to the transcript.
    pending_partial: str = ""


class SaidwireSTTProvider(WebSocketSTTProvider):
    """Saidwire streaming STT provider."""

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("saidwire_api_key")

    @property
    def name(self) -> str:
        return f"saidwire-{self._model}"

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name, vad_events_count=0)

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _SaidwireTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        return (
            f"{_WS_URL}"
            f"?model={self._model}"
            f"&sample_rate={sample_rate}"
            f"&encoding=linear16"
            f"&channels=1"
            f"&interim_results=true"
            f"&vad_events=true"
            f"&punctuate=true"
            # Native silence endpointing off; the final is forced via Finalize at
            # speech-end instead (TTFS parity with the Deepgram provider).
            f"&endpointing=false"
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._api_key.get_secret_value()}"}

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await self._send_and_wait(ws, json.dumps({"type": "Finalize"}), tx)
        await ws.send(json.dumps({"type": "CloseStream"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_SaidwireTranscription, tx)
        result = tx.result
        msg_type: str = msg.get("type", "")

        if msg_type == "SpeechStarted":
            if result.audio_start_time is not None and result.vad_first_detected is None:
                result.vad_first_detected = now - result.audio_start_time
                result.vad_first_event_content = str(msg)
            result.vad_events_count = (result.vad_events_count or 0) + 1
            return False

        if msg_type != "Results":
            return False

        transcript = _extract_transcript(msg)
        if not transcript:
            return False

        tx.partial(transcript, now)
        if msg.get("is_final"):
            tx.final(transcript, now)
            tx.pending_partial = ""
        elif len(transcript) > len(tx.pending_partial):
            tx.pending_partial = transcript
        return False

    def _finish(self, tx: Transcription) -> None:
        tx = cast(_SaidwireTranscription, tx)
        result = tx.result
        parts = tx.finals + ([tx.pending_partial] if tx.pending_partial else [])
        if parts:
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
