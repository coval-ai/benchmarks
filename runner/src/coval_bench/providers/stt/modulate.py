# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Modulate (Velma-2) real-time streaming STT provider.

Wire protocol: WebSocket, one endpoint per model; the model id is the endpoint
path segment (``velma-2-stt-streaming`` is multilingual,
``velma-2-stt-streaming-english-v2`` is fast English-only).
Auth: ``api_key`` query parameter (no header).
Audio in: raw binary PCM frames; format declared via query params
  (``audio_format=s16le&sample_rate=<hz>&num_channels=1``).
Close: an empty ``""`` text frame flushes the buffer; the server returns the
  final ``utterance`` message(s), then a ``done`` message, then closes.
Server messages (serialised JSON, keyed by ``type``):
  partial_utterance -> in-progress preview, ``partial_utterance.text``
  utterance         -> committed segment, ``utterance.text``
  done              -> stream complete, ``duration_ms``
  error             -> ``error`` description

The multilingual endpoint needs ``partial_results=true`` to emit partials and
runs with ``speaker_diarization=false`` so a single utterance is not split into
per-speaker finals; the finals it does emit are concatenated in arrival order.

The English endpoint emits partials on a fixed ~1.5 s cadence, so its TTFT
tracks the emission interval rather than engine latency and is excluded in
``registries/metrics.py``.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_BASE = "wss://platform.modulate.ai/api"
_EOS = ""

_MULTILINGUAL_MODEL = "velma-2-stt-streaming"
_ENGLISH_MODEL = "velma-2-stt-streaming-english-v2"


class ModulateSTTProvider(WebSocketSTTProvider):
    """Modulate streaming STT provider."""

    _PARTIAL_FALLBACK = "longest"

    def __init__(self, settings: Settings, model: str = _ENGLISH_MODEL) -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("modulate_api_key")

    @property
    def name(self) -> str:
        return f"modulate-{self._model}"

    def _ws_url(self, sample_rate: int) -> str:
        params: dict[str, str] = {
            "api_key": self._api_key.get_secret_value(),
            "audio_format": "s16le",
            "sample_rate": str(sample_rate),
            "num_channels": "1",
        }
        if self._model == _MULTILINGUAL_MODEL:
            params["partial_results"] = "true"
            params["speaker_diarization"] = "false"
        return f"{_WS_BASE}/{self._model}?{urlencode(params)}"

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await ws.send(_EOS)

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type: str = msg.get("type", "")

        if msg_type == "error":
            tx.fail(str(msg.get("error") or msg))
            self._log.warning("modulate_stt_error", msg=msg)
            return True

        if msg_type == "done":
            return True

        payload = msg.get(msg_type)
        transcript = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
        if not transcript:
            return False

        tx.partial(transcript, now)
        if msg_type == "utterance":
            tx.final(transcript, now)
        return False
