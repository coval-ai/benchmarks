# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Avartha real-time STT provider.

Avartha serves open models behind the ElevenLabs realtime protocol on its own
host, so the session and transcript events are the ElevenLabs adapter's. It is
stricter about the audio frames: each one carries sample_rate and commit. The
utterance is committed by hand and the session ends on its committed transcript.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import urlencode

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription
from coval_bench.providers.stt.elevenlabs import ElevenLabsSTTProvider

_WS_URL = "wss://api.preview.avartha.ai/inference/serverless/elevenlabs/v1/speech-to-text/realtime"


class AvarthaSTTProvider(ElevenLabsSTTProvider):
    """Avartha streaming STT provider."""

    _KEY_FIELD = "avartha_api_key"

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)

    @property
    def name(self) -> str:
        return f"avartha-{self._model}"

    def _ws_url(self, sample_rate: int) -> str:
        params = {
            "model_id": self._model,
            "audio_format": f"pcm_{sample_rate}",
            "commit_strategy": "manual",
        }
        # Qwen3-ASR can answer in another language without a hint; the other models refuse one.
        if "qwen" in self._model:
            params["language_code"] = "en"
        return f"{_WS_URL}?{urlencode(params, safe='/')}"

    def _frame(self, chunk: bytes, *, commit: bool) -> str:
        return json.dumps(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(chunk).decode("utf-8"),
                "sample_rate": self._SAMPLE_RATE,
                "commit": commit,
            }
        )

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(self._frame(chunk, commit=False))

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await ws.send(self._frame(b"", commit=True))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        if super()._on_message(msg, now, tx):
            return True
        return bool(msg.get("message_type") == "committed_transcript")
