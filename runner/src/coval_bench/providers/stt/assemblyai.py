# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""AssemblyAI real-time STT provider (v3 streaming API).

Models: universal-streaming, universal-streaming-multilingual, universal-N.M-pro
Wire protocol: WebSocket, wss://streaming.assemblyai.com/v3/ws
Auth: Authorization: <key>
Close: {"type": "Terminate"}
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlencode

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

# Maps user-facing model names to the speech_model value required by the API.
# format_turns is omitted (default=false) — the formatting pass adds latency that
# would inflate TTFT measurements.
_SPEECH_MODEL_MAP: dict[str, str] = {
    "universal-streaming": "universal-streaming-english",
    "universal-streaming-multilingual": "universal-streaming-multilingual",
}
# Pro releases (universal-3.5-pro, universal-3.6-pro, ...) need no code change: the
# registry id maps to the API's speech_model by swapping dots for dashes.
_PRO_MODEL = re.compile(r"^universal-\d+(?:\.\d+)?-pro$")
_WS_BASE = "wss://streaming.assemblyai.com/v3/ws"

# 1.0 = pure VAD silence-latency mode: the model never declares end-of-turn on a
# semantic guess, only on our ForceEndpoint at speech-end (TTFS parity). A lower
# value could let a confident short utterance auto-finalize before our signal.
_END_OF_TURN_CONFIDENCE_THRESHOLD = 1.0

# After ForceEndpoint, wait this long for the forced final before sending Terminate,
# so the close never races the final (the WS close-gate bug class). Falls through on
# timeout — the outer per-item timeout still bounds the run.

# Vendor-recommended voice-agent configuration for Pro models, applied on top of
# the base connection params. mode=min_latency is the documented accuracy/latency
# preset for voice agents (sets interruption_delay, turn-silence windows,
# continuous_partials, vad_threshold as a bundle):
# https://www.assemblyai.com/docs/streaming/prompting-and-keyterms
_PRO_EXTRA_PARAMS: dict[str, str] = {"mode": "min_latency"}


def _is_pro(model: str) -> bool:
    return _PRO_MODEL.match(model) is not None


def _speech_model(model: str) -> str:
    return _SPEECH_MODEL_MAP.get(model) or model.replace(".", "-")


class AssemblyAIProvider(WebSocketSTTProvider):
    """AssemblyAI v3 streaming STT provider."""

    _SAMPLE_RATE = 16000

    def _model_supported(self, model: str) -> bool:
        return model in _SPEECH_MODEL_MAP or _is_pro(model)

    def __init__(self, settings: Settings, model: str = "universal-streaming") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid AssemblyAI model {model!r}. "
                f"Valid: {sorted(_SPEECH_MODEL_MAP)} or universal-N.M-pro"
            )
        super().__init__(settings, model)
        self._api_key = self._require_secret("assemblyai_api_key")

    @property
    def name(self) -> str:
        return f"assemblyai-{self._model}"

    def _ws_url(self, sample_rate: int) -> str:
        url = f"{_WS_BASE}?sample_rate={sample_rate}&speech_model={_speech_model(self._model)}"
        if _is_pro(self._model):
            url += f"&end_of_turn_confidence_threshold={_END_OF_TURN_CONFIDENCE_THRESHOLD}"
            url += "&" + urlencode(_PRO_EXTRA_PARAMS)
        return url

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self._api_key.get_secret_value()}

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        if _is_pro(self._model):
            await self._send_and_wait(ws, json.dumps({"type": "ForceEndpoint"}), tx)
        await ws.send(json.dumps({"type": "Terminate"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type: str = msg.get("type", "")
        if msg_type in ("Begin", "Termination"):
            return False

        transcript = self._extract_transcript(msg)
        if transcript:
            # TTFT fires on the first Turn regardless of end_of_turn: time-to-first-word,
            # not time-to-first-completed-sentence.
            tx.partial(transcript, now)
        if msg_type == "Turn" and msg.get("end_of_turn") and transcript:
            tx.final(transcript, now)
        return False

    def _extract_transcript(self, msg: dict[str, Any]) -> str:
        if msg.get("type") != "Turn":
            return ""
        transcript = str(msg.get("transcript", "")).strip()
        if transcript:
            return transcript
        words: list[dict[str, Any]] = msg.get("words", [])
        parts = [
            str(w["text"]).strip()
            for w in words
            if isinstance(w, dict) and str(w.get("text", "")).strip()
        ]
        return " ".join(parts)
