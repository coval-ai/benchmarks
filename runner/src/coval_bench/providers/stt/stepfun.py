# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""StepFun realtime STT provider (WebSocket).

StepFun serves StepAudio ASR behind a Realtime-shaped bidirectional socket. The
session runs without ``turn_detection`` so the server performs no VAD: audio
streams in as base64 PCM16 appends and ``input_audio_buffer.commit`` finalizes
the buffer. Deltas carry the cumulative hypothesis (``text`` plus a correctable
``stash`` tail), not increments.
Protocol: https://platform.stepfun.ai/docs/en/api-reference/audio/asr-stream
"""

from __future__ import annotations

import json
from itertools import count
from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.stt._realtime import RealtimeSTTProvider
from coval_bench.providers.stt._streaming import Transcription

_WS_URL = "wss://api.stepfun.ai/v1/realtime/asr/stream"
_READY_TIMEOUT_S = 30.0
_DELTA = "conversation.item.input_audio_transcription.delta"
_COMPLETED = "conversation.item.input_audio_transcription.completed"


class StepfunSTTProvider(RealtimeSTTProvider):
    """StepFun streaming STT provider for the StepAudio ASR stream models."""

    _LABEL = "StepFun"
    _SAMPLE_RATE = 16000

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("stepfun_api_key")
        self._event_ids = count(1)

    @property
    def name(self) -> str:
        return "stepfun"

    def _ws_url(self, sample_rate: int) -> str:
        return _WS_URL

    def _frame(self, event_type: str, **fields: Any) -> str:
        return json.dumps(
            {"event_id": f"event_{next(self._event_ids)}", "type": event_type, **fields}
        )

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        session = {
            "audio": {
                "input": {
                    "format": {
                        "type": "pcm",
                        "codec": "pcm_s16le",
                        "rate": sample_rate,
                        "bits": 16,
                        "channel": 1,
                    },
                    "transcription": {"model": self._model, "language": "en"},
                }
            }
        }
        await ws.send(self._frame("session.update", session=session))
        await self._await_ready(ws, "session.updated", _READY_TIMEOUT_S)

    def _error_message(self, event: dict[str, Any]) -> str:
        error = event.get("error") or {}
        message = str(error.get("message") or "StepFun error")
        code = error.get("code")
        return f"{message} (code={code})" if code else message

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        event_type = str(msg.get("type", ""))

        if event_type == _DELTA:
            tx.partial(f"{msg.get('text', '')}{msg.get('stash', '')}", now)
            return False

        if event_type == _COMPLETED:
            transcript = str(msg.get("transcript", "")).strip()
            tx.partial(transcript, now)
            if msg.get("reason", "commit") != "commit":
                # An auto-split item is not the final; only the commit stamps it.
                if transcript:
                    tx.finals.append(transcript)
                return False
            tx.final(transcript, now)
            return True

        if event_type == "error":
            tx.fail(self._error_message(msg))
            self._log.warning("stepfun_stt_error", msg=msg)
            return True

        return False
