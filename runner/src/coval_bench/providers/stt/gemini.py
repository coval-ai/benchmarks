# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemini Live API STT provider (WebSocket) for ``gemini-3.5-transcribe-live``.

Distinct from the ``google`` provider, which speaks Cloud Speech v2 with
service-account auth; the Live API is plain JSON-over-WebSocket with API-key
auth. VERBATIM mode is pinned: SMART cleans disfluencies, which would skew WER
against our verbatim references.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
_SETUP_TIMEOUT_S = 30.0


def _decode_event(raw: str | bytes) -> dict[str, Any]:
    """The Live API sends JSON in binary frames; accept text frames too."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    event: dict[str, Any] = json.loads(text)
    return event


@dataclass
class _GeminiTranscription(Transcription):
    complete_seen: asyncio.Event = field(default_factory=asyncio.Event)


class GeminiSTTProvider(WebSocketSTTProvider):
    """Gemini Live API streaming transcription provider."""

    # The Live API sends JSON in binary frames.
    _SKIP_BINARY_FRAMES = False

    _PARTIAL_FALLBACK = "longest"
    _SAMPLE_RATE = 16000

    def __init__(self, settings: Settings, model: str = "gemini-3.5-transcribe-live") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("gemini_api_key")

    @property
    def name(self) -> str:
        return "gemini"

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _GeminiTranscription(result)

    def _build_setup(self) -> dict[str, Any]:
        return {
            "setup": {
                "model": f"models/{self._model}",
                "generationConfig": {"responseModalities": ["TEXT"]},
                "inputAudioTranscription": {
                    "languageCodes": ["en-US"],
                    "mode": "VERBATIM",
                },
            }
        }

    def _ws_url(self, sample_rate: int) -> str:
        return f"{_WS_URL}?key={self._api_key.get_secret_value()}"

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(json.dumps(self._build_setup()))
        try:
            async with asyncio.timeout(_SETUP_TIMEOUT_S):
                while True:
                    try:
                        raw = await ws.recv()
                    except StopAsyncIteration as exc:
                        raise RuntimeError(
                            f"Did not receive setupComplete within {_SETUP_TIMEOUT_S}s"
                        ) from exc
                    event = _decode_event(raw)
                    if "setupComplete" in event:
                        return
                    if "error" in event:
                        message = event["error"].get("message", "Unknown Gemini error")
                        raise RuntimeError(f"Gemini error during setup: {message}")
        except TimeoutError as exc:
            raise RuntimeError(f"Did not receive setupComplete within {_SETUP_TIMEOUT_S}s") from exc

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(
            json.dumps(
                {
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(chunk).decode("ascii"),
                            "mimeType": f"audio/pcm;rate={self._SAMPLE_RATE}",
                        }
                    }
                }
            )
        )

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        tx = cast(_GeminiTranscription, tx)
        # Events seen mid-audio belong to earlier utterances; only ones
        # arriving after this clear can be the stream-end flush.
        tx.final_event.clear()
        tx.complete_seen.clear()
        await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(self._FINAL_WAIT_S):
                await tx.final_event.wait()
                await tx.complete_seen.wait()
        await ws.close()

    def _decode(self, raw: str | bytes) -> Any:
        return _decode_event(raw)

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_GeminiTranscription, tx)
        if "error" in msg:
            tx.fail(str(msg["error"].get("message", "Gemini error")))
            self._log.warning("gemini_stt_error", msg=msg)
            return True

        server_content: dict[str, Any] = msg.get("serverContent") or {}

        interim_event = server_content.get("interimInputTranscription") or {}
        interim = str(interim_event.get("text", ""))
        if interim.strip():
            tx.partial(interim, now)

        final = str((server_content.get("inputTranscription") or {}).get("text", ""))
        if final.strip():
            tx.final(final, now)

        if server_content.get("generationComplete"):
            tx.complete_seen.set()
        return False
