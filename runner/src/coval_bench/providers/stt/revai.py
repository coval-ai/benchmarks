# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rev AI real-time streaming STT provider (Reverb model).

Wire protocol: WebSocket, wss://api.rev.ai/speechtotext/v1/stream
Auth: ``access_token`` query parameter.
Audio in: raw binary PCM frames.
Close: an ``"EOS"`` text frame; the server flushes the last final and closes.
Server messages (serialised JSON, keyed by ``type``):
  connected -> handshake ack, no transcript
  partial   -> best-guess hypothesis, ``elements: [{type: "text", value}]``
  final     -> committed segment, ``elements: [{type: "text"|"punct", value, ...}]``

``transcriber=machine_v2`` pins the Reverb ASR model rather than whatever
default is in effect for the account (see docs.rev.ai/api/streaming).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlencode

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider
from coval_bench.providers.stt._transcript_utils import finalize_transcript

_WS_BASE = "wss://api.rev.ai/speechtotext/v1/stream"
_TRANSCRIBER_FOR_MODEL: dict[str, str] = {"reverb": "machine_v2"}
_EOS = "EOS"

# Rev has no force-finalize: EOS discards the un-endpointed tail, so we feed
# trailing silence to make Reverb endpoint the last segment (cap; breaks early).
_EOT_SILENCE_S = 2.0


@dataclass
class _RevAITranscription(Transcription):
    # A partial left standing at close is the un-finalized tail; recover it.
    pending_partial: str = ""


class RevAISTTProvider(WebSocketSTTProvider):
    """Rev AI streaming STT provider."""

    _VALID_MODELS = frozenset(_TRANSCRIBER_FOR_MODEL)
    _PARTIAL_FALLBACK = "longest"

    def __init__(self, settings: Settings, model: str = "reverb") -> None:
        if not self._model_supported(model):
            raise ValueError(f"Invalid Rev AI model {model!r}. Valid: {sorted(self._VALID_MODELS)}")
        super().__init__(settings, model)
        self._api_key = self._require_secret("revai_api_key")

    @property
    def name(self) -> str:
        return "revai"

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _RevAITranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        params = {
            "access_token": self._api_key.get_secret_value(),
            "content_type": (
                f"audio/x-raw;layout=interleaved;rate={sample_rate};format=S16LE;channels=1"
            ),
            "language": "en",
            "transcriber": _TRANSCRIBER_FOR_MODEL[self._model],
        }
        return f"{_WS_BASE}?{urlencode(params)}"

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        tx = cast(_RevAITranscription, tx)
        # Clear first: a mid-stream final would leave the latch set and skip the
        # drain. Silence never sets audio_start_time (audio_to_final is from speech).
        tx.final_event.clear()
        silence = bytes(int(tx.byte_rate * _EOT_SILENCE_S))
        async for chunk, _ in paced_chunks(silence, tx.chunk_size, tx.byte_rate):
            if tx.final_event.is_set():
                break
            await ws.send(chunk)
        await ws.send(_EOS)

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_RevAITranscription, tx)
        msg_type: str = msg.get("type", "")
        if msg_type == "connected":
            return False

        transcript = self._elements_to_text(msg.get("elements", []))
        if not transcript:
            return False

        # TTFT is time-to-first-word: fire on the first partial, not the first final.
        tx.partial(transcript, now)
        if msg_type == "final":
            tx.final(transcript, now)
            tx.pending_partial = ""
        else:
            # Partials are cumulative per segment, so keep the latest — a
            # downward revision must win over a stale, longer earlier guess.
            tx.pending_partial = transcript
        return False

    def _finish(self, tx: Transcription) -> None:
        tx = cast(_RevAITranscription, tx)
        segments = tx.finals + ([tx.pending_partial] if tx.pending_partial else [])
        finalize_transcript(
            tx.result, final_segments=segments, partial_fallback=self._PARTIAL_FALLBACK
        )

    @staticmethod
    def _elements_to_text(elements: list[dict[str, Any]]) -> str:
        """Join the spoken-word (``text``) elements; ``punct`` is dropped for WER parity."""
        parts = [
            str(el.get("value", "")).strip()
            for el in elements
            if isinstance(el, dict)
            and el.get("type") == "text"
            and str(el.get("value", "")).strip()
        ]
        return " ".join(parts)
