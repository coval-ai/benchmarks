# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepgram real-time STT provider.

Supports models: default (nova-2), nova-2, nova-3, flux-general-en, flux-general-multi.
Wire protocols:
  nova-2 / nova-3 / default: wss://api.deepgram.com/v1/listen (Results events)
  flux-general-*:            wss://api.deepgram.com/v2/listen  (TurnInfo events)
Auth: Authorization: Token <key>
Close: {"type": "CloseStream"}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider


@dataclass
class _DeepgramTranscription(Transcription):
    # Longest interim not yet covered by a final; appended to the nova transcript.
    pending_partial: str = ""
    # Flux transcript is cumulative within a turn and resets per turn.
    flux_turns: dict[int, str] = field(default_factory=dict)


class DeepgramProvider(WebSocketSTTProvider):
    """Deepgram streaming STT provider."""

    def __init__(self, settings: Settings, model: str = "default") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("deepgram_api_key")

    @property
    def name(self) -> str:
        if self._model == "default":
            return "deepgram"
        return f"deepgram-{self._model}"

    @property
    def _is_flux(self) -> bool:
        return self._model.startswith("flux-")

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name, vad_events_count=0)

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _DeepgramTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        if self._is_flux:
            # v2/listen 400s on v1-only query params (channels/interim_results/no_delay).
            return (
                "wss://api.deepgram.com/v2/listen"
                f"?model={self._model}"
                f"&sample_rate={sample_rate}"
                f"&encoding=linear16"
                f"&eot_threshold=1.0"
                f"&eot_timeout_ms=60000"
            )
        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?sample_rate={sample_rate}"
            f"&encoding=linear16"
            f"&channels=1"
            f"&interim_results=true"
            f"&vad_events=true"
            f"&no_delay=true"
            f"&punctuate=true"
            f"&filler_words=true"
            # Disable native silence endpointing; we force the final at speech-end
            # via a Finalize message instead (TTFS parity).
            f"&endpointing=false"
        )
        if self._model != "default":
            url += f"&model={self._model}"
        return url

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._api_key.get_secret_value()}"}

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        if self._is_flux:
            tx.result.finalization_start_time = time.monotonic()
            if not await self._send_and_wait(ws, json.dumps({"type": "ForceEndTurn"}), tx):
                tx.result.finalization_timed_out = True
        else:
            # nova finalizes on our signal (endpointing disabled): wait for the forced
            # final before closing so the close can't race it.
            await self._send_and_wait(ws, json.dumps({"type": "Finalize"}), tx)
        await ws.send(json.dumps({"type": "CloseStream"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_DeepgramTranscription, tx)
        result = tx.result
        msg_type: str = msg.get("type", "")

        if msg_type == "SpeechStarted":
            if result.audio_start_time is not None and result.vad_first_detected is None:
                result.vad_first_detected = now - result.audio_start_time
                result.vad_first_event_content = str(msg)
            result.vad_events_count = (result.vad_events_count or 0) + 1
            return False

        if msg_type in ("UtteranceEnd", "Metadata", "Connected"):
            return False

        if msg_type == "Warning":
            code = str(msg.get("code", ""))
            if code == "FORCE_END_TURN_NO_ACTIVE_TURN":
                result.finalization_warning_code = code
                tx.final_event.set()
            return False

        if msg_type == "TurnInfo":
            transcript = str(msg.get("transcript", "")).strip() or _words_text(msg.get("words", []))
            if transcript:
                tx.partial(transcript, now)
                tx.flux_turns[int(msg.get("turn_index", 0))] = transcript
            if msg.get("event") == "EndOfTurn":
                result.finalization_trigger = str(msg.get("trigger", "")) or None
                audio_window_end = msg.get("audio_window_end")
                if isinstance(audio_window_end, (int, float)):
                    result.final_audio_window_end_seconds = float(audio_window_end)
                if result.finalization_start_time is not None:
                    result.finalization_latency_seconds = max(
                        0.0, now - result.finalization_start_time
                    )
                tx.final("", now)
            return False

        transcript = self._extract_transcript(msg)
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
        tx = cast(_DeepgramTranscription, tx)
        result = tx.result
        final_trigger = result.finalization_trigger
        if (
            self._is_flux
            and result.audio_to_final_seconds is not None
            and final_trigger != "manual"
            and result.error is None
        ):
            result.error = f"Flux EndOfTurn was not manually triggered (trigger={final_trigger!r})"

        if self._is_flux and final_trigger == "manual":
            joined = " ".join(tx.flux_turns[i] for i in sorted(tx.flux_turns)).strip()
            result.complete_transcript = joined or None
        elif tx.finals or tx.pending_partial:
            # Every is_final segment, plus the not-yet-finalized tail.
            parts = list(tx.finals)
            if tx.pending_partial:
                parts.append(tx.pending_partial)
            result.complete_transcript = " ".join(parts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())

    def _extract_transcript(self, msg: dict[str, Any]) -> str:
        # v1/listen Results: {"type": "Results", "channel": {"alternatives": [...]}}
        try:
            channel: dict[str, Any] = msg.get("channel", {})
            alternatives: list[dict[str, Any]] = channel.get("alternatives", [])
            if not alternatives:
                return ""
            alt = alternatives[0]
            return _words_text(alt.get("words", [])) or str(alt.get("transcript", "")).strip()
        except (KeyError, IndexError, TypeError):
            return ""


def _words_text(words: list[dict[str, Any]]) -> str:
    parts = []
    for w in words:
        text = str(w.get("punctuated_word", "")).strip() or str(w.get("word", "")).strip()
        if text:
            parts.append(text)
    return " ".join(parts)
