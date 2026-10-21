# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reson8 real-time STT provider (WebSocket)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlencode

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import (
    StreamAborted,
    Transcription,
    WebSocketSTTProvider,
)

_WS_PATH = "/v1/speech-to-text/realtime"
_WS_HOST = "wss://api.reson8.dev"
# Same model behind a US East edge; the row id carries the routing.
_GLOBAL_WS_HOST = "wss://api.global.reson8.dev"
_GLOBAL_SUFFIX = "-global"

# Pinned rather than auto-detected: the docs note detection is weaker on short
# utterances, and the STT corpus is English.
_LANGUAGE = "en"


_FLUSH_ID = "eos"

# Documented error codes. Reson8 publishes the codes but not the frame carrying
# them, so _error_text tests for non-success rather than one literal shape.
_ERROR_CODES = frozenset(
    {"INVALID_REQUEST", "UNAUTHORIZED", "NOT_FOUND", "PAYLOAD_TOO_LARGE", "INTERNAL_ERROR"}
)


def _error_text(msg: dict[str, Any]) -> str | None:
    """Failure description when *msg* is an error frame, else ``None``."""
    frames = [msg]
    for key in ("error", "details"):
        nested = msg.get(key)
        if isinstance(nested, dict):
            frames.append(nested)

    for frame in frames:
        code = frame.get("code")
        if str(frame.get("type", "")).casefold() != "error" and not (
            isinstance(code, str) and code.upper() in _ERROR_CODES
        ):
            continue
        detail = frame.get("message") or frame.get("detail") or code
        return str(detail) if detail else "Reson8 STT error"

    error = msg.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    return None


@dataclass
class _Reson8Transcription(Transcription):
    # Set when the flush is acked or the receiver ends; the sender closes on it.
    flushed: asyncio.Event = field(default_factory=asyncio.Event)
    flush_confirmed: bool = False
    recent_frames: list[str] = field(default_factory=list)


class Reson8STTProvider(WebSocketSTTProvider):
    """Reson8 streaming STT provider."""

    _PARTIAL_FALLBACK = "longest"
    _NO_FINAL_ERROR = "ws_closed_without_final_transcript"

    def __init__(self, settings: Settings, model: str = "realtime") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("reson8_api_key")
        host = _GLOBAL_WS_HOST if model.endswith(_GLOBAL_SUFFIX) else _WS_HOST
        self._ws_base = host + _WS_PATH

    @property
    def name(self) -> str:
        return "reson8"

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _Reson8Transcription(result)

    def _ws_url(self, sample_rate: int) -> str:
        """Connection URL. Auth rides a header, never a query parameter."""
        params = {
            # ``auto`` sniffs container headers; our frames are header-less PCM.
            "encoding": "pcm_s16le",
            "sample_rate": str(sample_rate),
            "channels": "1",
            "language": _LANGUAGE,
            # Without this the server returns finals only, and no ``is_final`` flag.
            "include_interim": "true",
        }
        return f"{self._ws_base}?{urlencode(params)}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"ApiKey {self._api_key.get_secret_value()}"}

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        tx = cast(_Reson8Transcription, tx)
        # No end-of-stream message exists: the client closes once the flush is
        # acknowledged, which also releases the receive loop.
        frame = json.dumps({"type": "flush_request", "id": _FLUSH_ID})
        await self._send_and_wait(ws, frame, tx, event=tx.flushed)
        await ws.close()

    async def _receive(self, ws: Any, tx: Transcription) -> None:
        # Wraps the template: a receive failure fails the row unless the flush was
        # acked, even after a segment final (the template gates on audio_to_final).
        tx = cast(_Reson8Transcription, tx)
        try:
            await super()._receive(ws, tx)
        except StreamAborted:
            raise
        except Exception as exc:
            if not tx.flush_confirmed:
                tx.fail(str(exc))
            raise
        finally:
            tx.flushed.set()

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_Reson8Transcription, tx)
        tx.recent_frames.append(str(msg)[:200])
        del tx.recent_frames[:-3]

        error = _error_text(msg)
        if error is not None:
            tx.fail(error)
            self._log.warning("reson8_stt_error", msg=msg)
            return True

        msg_type = str(msg.get("type", ""))
        if msg_type == "flush_confirmation":
            tx.flush_confirmed = True
            return True
        if msg_type != "transcript":
            return False

        transcript = str(msg.get("text", "")).strip()
        if not transcript:
            return False
        # Absent means the server is returning finals only, which is its
        # behaviour when include_interim is off.
        if msg.get("is_final", True):
            tx.final(transcript, now)
        else:
            tx.partial(transcript, now)
        return False

    def _finish(self, tx: Transcription) -> None:
        tx = cast(_Reson8Transcription, tx)
        if tx.result.error is None and not tx.flush_confirmed:
            # Not a failure on its own — finals may already be in — but the tail
            # went unflushed, so the transcript can be short.
            self._log.warning(
                "reson8_flush_unconfirmed",
                had_final=bool(tx.finals),
                recent_frames=tx.recent_frames,
            )
        super()._finish(tx)
