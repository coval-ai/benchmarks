# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Together AI real-time STT provider (WebSocket).

Wire protocol (OpenAI-Realtime style):
  URL:   wss://api.together.ai/v1/realtime?model=<id>&input_audio_format=pcm_s16le_16000
  Auth:  Authorization: Bearer <key> + OpenAI-Beta: realtime=v1
  Send:  {"type": "input_audio_buffer.append", "audio": <base64 PCM>} then
         {"type": "input_audio_buffer.commit"} to force the final.
  Recv:  conversation.item.input_audio_transcription.delta (partial) /
         .completed (final). The server never closes; the client closes after
         draining the forced final.

The Nemotron models run with ~1.1 s of encoder lookahead and commit drops it,
truncating the tail. Trailing silence is paced in before the commit to flush
the lookahead (Rev AI pattern); their TTFS is excluded in
``registries/metrics.py`` since the final's timing then tracks the client's
silence length, not the engine.

Parakeet's endpointer can close a segment with an empty ``completed`` even
though the segment's deltas carried the text; such finals count as real
finalizations, with the item's latest delta salvaged as the segment text
(punctuation-only remnants excepted).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider
from coval_bench.providers.stt._transcript_utils import finalize_transcript

_MODEL_IDS = {
    "nemotron-3-asr-streaming-0.6b": "nvidia/nemotron-3-asr-streaming-0.6b",
    "nemotron-3.5-asr-streaming-0.6b": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "whisper-large-v3": "openai/whisper-large-v3",
}

# Models whose commit drops the encoder-lookahead buffer; they get trailing
# silence before the commit so the tail is decoded.
_LOOKAHEAD_FLUSH_MODELS = frozenset(
    {"nemotron-3-asr-streaming-0.6b", "nemotron-3.5-asr-streaming-0.6b"}
)

# Nemotron's deepest lookahead config is 1.12 s; pad past it.
_TAIL_SILENCE_S = 1.6


@dataclass
class _TogetherTranscription(Transcription):
    # Deltas are cumulative running transcripts revised in place, keyed by
    # conversation item — keep only each item's latest.
    item_deltas: dict[str, str] = field(default_factory=dict)


class TogetherSTTProvider(WebSocketSTTProvider):
    """Together AI streaming STT provider."""

    # Nemotron's forced final can take a while after commit.
    _FINAL_WAIT_S = 15.0

    _VALID_MODELS = frozenset(_MODEL_IDS)
    _SAMPLE_RATE = 16000

    def __init__(self, settings: Settings, model: str = "whisper-large-v3") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Together STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        super().__init__(settings, model)
        self._api_key = self._require_secret("together_api_key")

    @property
    def name(self) -> str:
        return f"together-{self._model}"

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _TogetherTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        return (
            "wss://api.together.ai/v1/realtime"
            f"?model={_MODEL_IDS[self._model]}"
            "&input_audio_format=pcm_s16le_16000"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "OpenAI-Beta": "realtime=v1",
        }

    def _prepare_audio(self, audio_data: bytes, sample_rate: int) -> tuple[bytes, int]:
        byte_rate = sample_rate * 2
        if self._model in _LOOKAHEAD_FLUSH_MODELS:
            audio_data += bytes(int(byte_rate * _TAIL_SILENCE_S))
        return audio_data, byte_rate

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await self._send_and_wait(ws, json.dumps({"type": "input_audio_buffer.commit"}), tx)
        # The server holds the socket open; close so the receive loop ends.
        await ws.close()

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_TogetherTranscription, tx)
        msg_type: str = msg.get("type", "")

        if msg_type == "error":
            error_obj = msg.get("error")
            detail = (error_obj.get("message") if isinstance(error_obj, dict) else None) or msg.get(
                "message"
            )
            tx.fail(str(detail or msg))
            self._log.warning("together_stt_error", msg=msg)
            return True

        if msg_type.endswith("input_audio_transcription.delta"):
            delta = str(msg.get("delta", ""))
            if delta.strip():
                tx.partial(delta, now)
                tx.item_deltas[str(msg.get("item_id", ""))] = delta
            return False

        if msg_type.endswith("input_audio_transcription.completed"):
            transcript = str(msg.get("transcript", "")).strip()
            if not transcript:
                salvaged = tx.item_deltas.get(str(msg.get("item_id", "")), "").strip()
                if any(ch.isalnum() for ch in salvaged):
                    transcript = salvaged
            if transcript:
                tx.final(transcript, now)
            else:
                tx.final_event.set()
        return False

    def _finish(self, tx: Transcription) -> None:
        tx = cast(_TogetherTranscription, tx)
        segments = tx.finals or [d.strip() for d in tx.item_deltas.values()]
        finalize_transcript(tx.result, final_segments=segments, partial_fallback="none")
