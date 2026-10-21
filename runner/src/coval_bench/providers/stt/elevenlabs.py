# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs real-time STT provider (scribe_v2_realtime).

Wire protocol: WebSocket, wss://api.elevenlabs.io/v1/speech-to-text/realtime
Auth: xi-api-key: <key>
Audio: base64-encoded PCM chunks, JSON-wrapped
Close: {"message_type": "input_audio_chunk", "audio_base_64": "", "commit": true}
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_SESSION_START_TIMEOUT_S = 5.0

_ERROR_MSG_TYPES = frozenset(
    {
        "error",
        "auth_error",
        "quota_exceeded",
        "commit_throttled",
        "rate_limited",
        "unaccepted_terms",
        "queue_overflow",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "input_error",
        "chunk_size_exceeded",
        "insufficient_audio_activity",
        "transcriber_error",
        "scribe_error",
        "scribe_auth_error",
        "scribe_quota_exceeded_error",
        "scribe_throttled_error",
        "scribe_unaccepted_terms_error",
        "scribe_rate_limited_error",
        "scribe_queue_overflow_error",
        "scribe_resource_exhausted_error",
        "scribe_session_time_limit_exceeded_error",
        "scribe_input_error",
        "scribe_chunk_size_exceeded_error",
        "scribe_insufficient_audio_activity_error",
        "scribe_transcriber_error",
    }
)


class ElevenLabsSTTProvider(WebSocketSTTProvider):
    """ElevenLabs real-time STT provider."""

    _SAMPLE_RATE = 16000
    # No committed transcript: score the last partial rather than fail the clip.
    _PARTIAL_FALLBACK = "last"

    def __init__(self, settings: Settings, model: str = "scribe_v2_realtime") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("elevenlabs_api_key")

    @property
    def name(self) -> str:
        return f"elevenlabs-{self._model}"

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name, vad_events_count=0)

    def _ws_url(self, sample_rate: int) -> str:
        return (
            f"wss://api.elevenlabs.io/v1/speech-to-text/realtime"
            f"?model_id={self._model}&audio_format=pcm_{sample_rate}"
        )

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key.get_secret_value()}

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=_SESSION_START_TIMEOUT_S)
        except TimeoutError:
            self._log.warning("elevenlabs_session_started_timeout")
            return
        session: dict[str, Any] = json.loads(raw)
        if session.get("message_type") != "session_started":
            self._log.warning("elevenlabs_unexpected_first_message", msg=session)

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        b64 = base64.b64encode(chunk).decode("utf-8")
        await ws.send(json.dumps({"message_type": "input_audio_chunk", "audio_base_64": b64}))

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await ws.send(
            json.dumps({"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True})
        )

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type: str = msg.get("message_type", "")

        if msg_type in _ERROR_MSG_TYPES:
            error_text = msg.get("message", msg.get("error", "Unknown error"))
            tx.fail(f"{msg_type}: {error_text}")
            self._log.warning("elevenlabs_api_error", msg_type=msg_type, error=error_text)
            return True

        if msg_type == "partial_transcript":
            tx.partial(str(msg.get("text", "")), now)
        elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
            # The _with_timestamps variant only arrives with include_timestamps=true,
            # but it carries "text" in the same place, so accept it rather than drop it.
            text = str(msg.get("text", "")).strip()
            if text:
                tx.final(text, now)
        return False
