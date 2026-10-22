# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Speechmatics real-time STT provider.

Supports models: default, enhanced, broadcast, linden-1 (Agent STT preview).
Wire protocol: WebSocket, wss://global.rt.speechmatics.com/v2
Auth: Authorization: Bearer <key>
Start: {"message": "StartRecognition", ...}
Close: {"message": "EndOfStream", "last_seq_no": N}
"""

from __future__ import annotations

import json
from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = "wss://global.rt.speechmatics.com/v2"
_AGENT_WS_URL = "wss://preview.rt.speechmatics.com/v2/agent"


class SpeechmaticsProvider(WebSocketSTTProvider):
    """Speechmatics real-time STT provider."""

    def __init__(self, settings: Settings, model: str = "default") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("speechmatics_api_key")
        self._is_agent = model == "linden-1"
        # AddAudio count for EndOfStream; one clip per instance, reset on each _open.
        self._seq_no = 0

    @property
    def name(self) -> str:
        if self._model == "default":
            return "speechmatics"
        return f"speechmatics-{self._model}"

    def _build_start_recognition_config(self, sample_rate: int) -> dict[str, Any]:
        transcription_config: dict[str, Any] = {
            "language": "en",
            "enable_partials": True,
        }
        if self._model == "enhanced":
            transcription_config["operating_point"] = "enhanced"
        elif self._model == "broadcast":
            transcription_config["domain"] = "broadcast"
        elif self._is_agent:
            # Required: without it the /agent endpoint falls back to the RT API.
            transcription_config["model"] = self._model

        return {
            "message": "StartRecognition",
            "transcription_config": transcription_config,
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
        }

    def _ws_url(self, sample_rate: int) -> str:
        return _AGENT_WS_URL if self._is_agent else _WS_URL

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        self._seq_no = 0
        await ws.send(json.dumps(self._build_start_recognition_config(sample_rate)))
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            msg: dict[str, Any] = json.loads(raw)
            if msg.get("message") == "RecognitionStarted":
                return
            if msg.get("message") == "Error":
                raise RuntimeError(f"Speechmatics error: {msg.get('reason', 'Unknown error')}")

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(chunk)
        self._seq_no += 1

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": self._seq_no}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type: str = msg.get("message", "")
        if msg_type in ("AudioAdded", "Warning", "Info"):
            return False
        if msg_type == "EndOfTranscript":
            return True
        if msg_type == "Error":
            reason: str = msg.get("reason", "Unknown error")
            if "EndOfStream" not in reason and "schema" not in reason.lower():
                tx.fail(reason)
            return False

        transcript = self._extract_transcript(msg)
        if transcript:
            tx.partial(transcript, now)
        final_type = "AddSegment" if self._is_agent else "AddTranscript"
        if msg_type == final_type:
            tx.final(transcript, now)
        return False

    def _extract_transcript(self, msg: dict[str, Any]) -> str:
        msg_type: str = msg.get("message", "")
        if msg_type in ("AddSegment", "AddPartialSegment"):
            segment: dict[str, Any] = msg.get("segment", {})
            return str(segment.get("transcript", "")).strip()
        if msg_type not in ("AddTranscript", "AddPartialTranscript"):
            return ""
        # Prefer the pre-formatted transcript field — it handles punctuation
        # spacing correctly via attaches_to semantics (e.g. "Hello, world."
        # not "Hello , world .").
        transcript: str = msg.get("transcript", "").strip()
        if transcript:
            return transcript
        # Fallback for responses that omit the transcript field
        results: list[dict[str, Any]] = msg.get("results", [])
        parts: list[str] = []
        for r in results:
            alternatives: list[dict[str, Any]] = r.get("alternatives", [])
            if alternatives:
                content = alternatives[0].get("content", "")
                if content:
                    parts.append(content)
        return " ".join(parts).strip()
