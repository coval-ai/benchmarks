# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Avartha TTS provider.

Avartha serves open models behind the ElevenLabs stream-input protocol on its
own host. It only runs in auto mode, and a flush asks for the audio. Drama
answers repeated text from a stored render unless fresh_take asks for a live
one; the other models refuse the parameter.
"""

from __future__ import annotations

import json

from coval_bench.providers.tts.elevenlabs import _OUTPUT_FORMAT, ElevenLabsTTSProvider

_WS_URL = (
    "wss://api.preview.avartha.ai/inference/serverless/elevenlabs/v1"
    "/text-to-speech/{voice_id}/stream-input"
)


class AvarthaTTSProvider(ElevenLabsTTSProvider):
    """Avartha TTS provider over the single-context stream-input WebSocket."""

    _PROVIDER = "avartha"
    _KEY_FIELD = "avartha_api_key"

    def _session_frames(self, text: str) -> tuple[str, list[str], list[str]]:
        url = (
            f"{_WS_URL.format(voice_id=self._voice)}"
            f"?model_id={self._model}&output_format={_OUTPUT_FORMAT}&auto_mode=true"
        )
        if "drama" in self._model:
            url += "&fresh_take=true"
        setup = [json.dumps({"text": " "})]
        inputs = [json.dumps({"text": text, "flush": True}), json.dumps({"text": ""})]
        return url, setup, inputs
