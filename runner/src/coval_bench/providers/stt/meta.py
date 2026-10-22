# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Meta Muse Voice Transcribe real-time STT provider.

  wss://api.meta.ai/v1/asr/realtime
Auth: in the first JSON frame ({"authorization": {"accessToken": "Bearer <key>"}});
the Authorization header is ignored. Server acks with {"sessionId": ...}.
Finalize: {"type": "endStream"}, then read until the server closes with 1000.
"""

from __future__ import annotations

import json
from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider
from coval_bench.providers.stt._transcript_utils import finalize_transcript

_WS_URL = "wss://api.meta.ai/v1/asr/realtime"
_ENCODINGS = {16000: "PCM_16KHZ", 24000: "PCM_24KHZ"}


class MetaSTTProvider(WebSocketSTTProvider):
    """Meta Muse Voice Transcribe streaming STT."""

    _NO_FINAL_ERROR = "Muse stream closed without a final transcript"

    def __init__(self, settings: Settings, model: str = "muse-voice-transcribe-1.0") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("meta_api_key")

    @property
    def name(self) -> str:
        return "meta"

    def _ws_url(self, sample_rate: int) -> str:
        # Two accepted rates, so the base's single _SAMPLE_RATE guard can't express it.
        if sample_rate not in _ENCODINGS:
            raise ValueError(f"Muse Voice Transcribe needs 16 or 24 kHz PCM; got {sample_rate} Hz")
        return _WS_URL

    def _handshake(self, sample_rate: int) -> dict[str, Any]:
        return {
            "authorization": {"accessToken": f"Bearer {self._api_key.get_secret_value()}"},
            "audioEncoding": _ENCODINGS[sample_rate],
            "model": self._model,
            "mode": "PUSH_TO_TALK",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
            "languageBias": ["English"],
        }

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(json.dumps(self._handshake(sample_rate)))
        ack: dict[str, Any] = json.loads(await ws.recv())
        if "sessionId" not in ack:
            raise RuntimeError(f"Muse handshake rejected: {ack.get('message', ack)}")

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await ws.send(json.dumps({"type": "endStream"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type = msg.get("type")
        if msg_type == "error":
            tx.fail(str(msg.get("message") or "Muse stream error"))
            self._log.warning("meta_stt_error", msg=msg)
            return True
        if msg_type != "transcript":
            return False

        text = str(msg.get("transcript", "")).strip()
        tx.partial(text, now)
        if msg.get("final"):
            tx.final(text, now)
        return False

    def _finish(self, tx: Transcription) -> None:
        # Partials are cumulative, so the last final is the whole transcript.
        finalize_transcript(tx.result, explicit_transcript=tx.finals[-1] if tx.finals else None)
