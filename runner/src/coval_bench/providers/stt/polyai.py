# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""PolyAI Owl STT provider (WebSocket).

Owl is served behind an OpenAI-Realtime-shaped transcription socket, so the
session, audio framing and transcript events are the OpenAI adapter's; only
the host, the key and the model differ.
"""

from __future__ import annotations

from coval_bench.config import Settings
from coval_bench.providers.stt.openai import OpenAISTTProvider


class PolyAISTTProvider(OpenAISTTProvider):
    """PolyAI streaming STT provider for the Owl models."""

    _LABEL = "PolyAI"
    _URL = "wss://owl.polyai-tools.com/v1/realtime?intent=transcription"
    _KEY_FIELD = "polyai_owl_api_key"

    def __init__(self, settings: Settings, model: str = "dialog-owl-1.0") -> None:
        super().__init__(settings, model)

    @property
    def name(self) -> str:
        return f"polyai-{self._model}"
