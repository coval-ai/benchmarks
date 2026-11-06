# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fish Audio realtime STT provider.

Protocol: one MessagePack map per binary frame in both directions on
``wss://api.fish.audio/v1/asr/realtime``; the model is a ``model`` header.
Turn detection runs in manual mode and the clip's turn is closed with
``finalize``, then ``audio_done`` ends the session.
"""

from __future__ import annotations

from typing import Any

import ormsgpack

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import (
    StreamAborted,
    Transcription,
    WebSocketSTTProvider,
)

_WS_URL = "wss://api.fish.audio/v1/asr/realtime"


def _pack(msg: dict[str, Any]) -> bytes:
    return ormsgpack.packb(msg)


class FishAudioSTTProvider(WebSocketSTTProvider):
    """Fish Audio streaming STT provider."""

    _SKIP_BINARY_FRAMES = False

    def __init__(self, settings: Settings, model: str = "transcribe-stream-1") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("fishaudio_api_key")

    @property
    def name(self) -> str:
        return f"fishaudio-{self._model}"

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name, vad_events_count=0)

    def _ws_url(self, sample_rate: int) -> str:
        return _WS_URL

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "model": self._model,
        }

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            _pack(
                {
                    "type": "config",
                    "audio": {"format": "pcm_s16le", "sample_rate": sample_rate, "channels": 1},
                    "turn_detection": {"mode": "manual"},
                    # Speaker labels and event tags would land in the scored text.
                    "results": {"drafts": True, "diarize": False, "tag_audio_events": False},
                }
            )
        )
        while True:
            msg = self._decode(await ws.recv())
            if msg.get("type") == "session_started":
                return
            if msg.get("type") == "error":
                self._on_message(msg, 0.0, tx)
                raise StreamAborted(tx.result.error)

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(_pack({"type": "audio", "data": chunk}))

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await self._send_and_wait(ws, _pack({"type": "finalize"}), tx)
        await ws.send(_pack({"type": "audio_done"}))

    def _decode(self, raw: str | bytes) -> Any:
        return ormsgpack.unpackb(raw.encode() if isinstance(raw, str) else raw)

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        result = tx.result
        msg_type: str = msg.get("type", "")

        if msg_type == "speech_start":
            if result.audio_start_time is not None and result.vad_first_detected is None:
                result.vad_first_detected = now - result.audio_start_time
                result.vad_first_event_content = str(msg)
            result.vad_events_count = (result.vad_events_count or 0) + 1
        elif msg_type == "draft":
            text = str(msg.get("text") or "").strip()
            if text:
                tx.partial(text, now)
        elif msg_type == "committed":
            tx.final(str(msg.get("text") or "").strip(), now)
        elif msg_type == "error":
            self._log.warning("fishaudio_stt_error", msg=msg)
            if not msg.get("recoverable"):
                tx.fail(str(msg.get("message") or f"Fish Audio STT error ({msg.get('code')})"))
                return True
        return msg_type == "session_done"
