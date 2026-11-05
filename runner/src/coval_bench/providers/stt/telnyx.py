# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Telnyx real-time STT provider (WebSocket).

Telnyx fronts a dozen engines, its own self-hosted Whisper and Parakeet among
them, behind one socket:

  URL:  wss://api.telnyx.com/v2/speech-to-text/transcription
        ?transcription_engine=<engine>&model=<id>&input_format=linear16
        &sample_rate=<hz>&language=en-US&interim_results=true
  Auth: Authorization: Bearer <api key>
  Send: binary PCM frames.
  Recv: {"transcript": "...", "is_final": bool, "confidence": float}
        {"errors": [{"code", "title", "detail"}]} then the socket closes.

The model id is passed through as-is; the engine is derived from its vendor
prefix (``openai/whisper-tiny`` is Telnyx's own Whisper serving,
``nvidia/parakeet-v3`` is Parakeet, ``deepgram/nova-3`` is Deepgram, and so
on). An unprefixed id is a Deepgram model, the API's default engine.
``GET /v2/speech-to-text/providers?service_type=streaming`` lists what the
socket accepts; the larger Whisper is file-only despite the docs.

Whisper and Parakeet send no interims, only one final a few seconds after the
audio ends, so their TTFT equals audio-to-final.

Deepgram, Speechmatics and Soniox flush on ``{"type": "CloseStream"}`` and
close the socket themselves. The other engines have no end-of-audio frame; the
client waits for the final and closes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = "wss://api.telnyx.com/v2/speech-to-text/transcription"

# Vendor prefix of the model id -> Telnyx `transcription_engine` value.
_ENGINE_BY_VENDOR = {
    "deepgram": "Deepgram",
    "openai": "Telnyx",
    "nvidia": "Parakeet",
    "omi-health": "Parakeet",
    "azure": "Azure",
    "xai": "xAI",
    "assemblyai": "AssemblyAI",
    "speechmatics": "Speechmatics",
    "soniox": "Soniox",
    "cohere": "Cohere",
    "reson8": "Reson8",
    "humain": "Humain",
}
_DEFAULT_ENGINE = "Deepgram"
_CLOSE_STREAM_ENGINES = frozenset({"Deepgram", "Speechmatics", "Soniox"})


def engine_for(model: str) -> str:
    vendor, sep, _ = model.partition("/")
    if not sep:
        return _DEFAULT_ENGINE
    try:
        return _ENGINE_BY_VENDOR[vendor]
    except KeyError:
        raise ValueError(
            f"Unknown Telnyx STT vendor prefix {vendor!r} in model {model!r}. "
            f"Known: {sorted(_ENGINE_BY_VENDOR)}"
        ) from None


def _error_text(errors: Any) -> str:
    parts = []
    for err in errors if isinstance(errors, list) else []:
        if not isinstance(err, dict):
            continue
        bits = [str(err[k]) for k in ("code", "title", "detail") if err.get(k)]
        if bits:
            parts.append(" ".join(bits))
    return "; ".join(parts) or f"Telnyx STT error: {errors}"


@dataclass
class _TelnyxTranscription(Transcription):
    # Longest interim not yet covered by a final; appended to the transcript.
    pending_partial: str = ""


class TelnyxSTTProvider(WebSocketSTTProvider):
    """Streaming STT through Telnyx's multi-engine transcription socket."""

    _FINAL_WAIT_S = 10.0

    def __init__(self, settings: Settings, model: str = "openai/whisper-tiny") -> None:
        self._engine = engine_for(model)
        super().__init__(settings, model)
        self._api_key = self._require_secret("telnyx_api_key")

    @property
    def name(self) -> str:
        return f"telnyx-{self._model}"

    @property
    def engine(self) -> str:
        return self._engine

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _TelnyxTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        return (
            f"{_WS_URL}"
            f"?transcription_engine={self._engine}"
            f"&model={self._model}"
            f"&input_format=linear16"
            f"&sample_rate={sample_rate}"
            f"&language=en-US"
            f"&interim_results=true"
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        if self._engine in _CLOSE_STREAM_ENGINES:
            await self._send_and_wait(ws, json.dumps({"type": "CloseStream"}), tx)
            return
        tx.final_event.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(tx.final_event.wait(), timeout=self._FINAL_WAIT_S)
        await ws.close()

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_TelnyxTranscription, tx)
        if not isinstance(msg, dict):
            return False

        if "errors" in msg:
            tx.fail(_error_text(msg["errors"]))
            self._log.warning("telnyx_stt_error", msg=msg)
            return True

        transcript = str(msg.get("transcript") or "").strip()
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
        tx = cast(_TelnyxTranscription, tx)
        result = tx.result
        parts = tx.finals + ([tx.pending_partial] if tx.pending_partial else [])
        if parts:
            result.complete_transcript = " ".join(parts).strip() or None
        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
