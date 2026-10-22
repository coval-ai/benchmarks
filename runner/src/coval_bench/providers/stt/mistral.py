# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Mistral real-time STT provider (Voxtral realtime transcription).

Wire protocol: WebSocket, wss://api.mistral.ai/v1/audio/transcriptions/realtime
Auth: Authorization: Bearer <key>
Close: {"type": "input_audio.flush"} + {"type": "input_audio.end"}, server
replies with transcription.done
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider
from coval_bench.providers.stt._transcript_utils import finalize_transcript, set_first_token

_WS_BASE = "wss://api.mistral.ai/v1/audio/transcriptions/realtime"
_DEFAULT_MODEL = "voxtral-mini-transcribe-realtime-2602"

# Sample rates accepted by the realtime endpoint for pcm_s16le input.
_SUPPORTED_SAMPLE_RATES = frozenset({8000, 16000, 22050, 44100, 48000})

_SESSION_CREATED_TIMEOUT_S = 10.0


@dataclass
class _MistralTranscription(Transcription):
    # Raw deltas, spaces included; their concatenation is the fallback transcript.
    deltas: list[str] = field(default_factory=list)


class MistralSTTProvider(WebSocketSTTProvider):
    """Mistral Voxtral realtime transcription provider."""

    def __init__(self, settings: Settings, model: str = _DEFAULT_MODEL) -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("mistral_api_key")

    @property
    def name(self) -> str:
        return f"mistral-{self._model}"

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _MistralTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        # Checked here so an unsupported rate fails before any socket is opened.
        if sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Mistral realtime does not support {sample_rate} Hz; "
                f"supported: {sorted(_SUPPORTED_SAMPLE_RATES)}"
            )
        return f"{_WS_BASE}?model={self._model}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await self._await_session_created(ws)
        # The audio format is rejected once audio has started, so it is set
        # before the first append. target_streaming_delay_ms is left at the
        # server default so latency reflects out-of-box behaviour.
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "audio_format": {"encoding": "pcm_s16le", "sample_rate": sample_rate}
                    },
                }
            )
        )

    async def _await_session_created(self, ws: Any) -> None:
        try:
            async with asyncio.timeout(_SESSION_CREATED_TIMEOUT_S):
                while True:
                    try:
                        raw = await ws.recv()
                    except StopAsyncIteration:
                        raise RuntimeError("Mistral stream closed before session.created") from None
                    if isinstance(raw, bytes):
                        continue
                    msg: dict[str, Any] = json.loads(raw)
                    msg_type = msg.get("type", "")
                    if msg_type == "session.created":
                        return
                    if msg_type == "error":
                        raise RuntimeError(self._error_message(msg))
        except TimeoutError as exc:
            raise TimeoutError("Timed out waiting for session.created") from exc

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(
            json.dumps(
                {"type": "input_audio.append", "audio": base64.b64encode(chunk).decode("ascii")}
            )
        )

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # Flush forces any buffered audio through the model, end signals
        # end-of-stream; the server answers with transcription.done.
        await ws.send(json.dumps({"type": "input_audio.flush"}))
        await ws.send(json.dumps({"type": "input_audio.end"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_MistralTranscription, tx)
        msg_type: str = msg.get("type", "")

        if msg_type == "transcription.text.delta":
            text = str(msg.get("text", ""))
            if text:
                # A standalone " " delta is a word separator and must reach the
                # joined fallback transcript; it never stamps TTFT or a partial.
                tx.deltas.append(text)
            if text.strip():
                set_first_token(tx.result, text, now=now)
                tx.result.partial_transcripts.append(text)
            return False

        if msg_type == "transcription.done":
            text = str(msg.get("text", "")).strip()
            # A done with no text after a delta-less session is a dead stream,
            # not a final.
            if text or tx.deltas:
                tx.final(text, now)
            else:
                tx.fail("Mistral session produced an empty transcript")
            return True

        if msg_type == "error":
            tx.fail(self._error_message(msg))
            self._log.warning("mistral_stt_error", msg=msg)
            return True

        return False

    def _finish(self, tx: Transcription) -> None:
        tx = cast(_MistralTranscription, tx)
        finalize_transcript(
            tx.result,
            final_segments=tx.finals or ["".join(tx.deltas)],
            partial_fallback="none",
            join="",
        )

    def _error_message(self, msg: dict[str, Any]) -> str:
        detail = msg.get("error")
        if isinstance(detail, dict):
            message = detail.get("message")
            if isinstance(message, str) and message:
                return message
            if isinstance(message, dict):
                inner = message.get("detail")
                if isinstance(inner, str) and inner:
                    return inner
        return "Mistral realtime transcription error"
