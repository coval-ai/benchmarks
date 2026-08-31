# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas TTS provider — WebSocket streaming.

TTFA is measured from the text submit to the first PCM frame; session setup
(connect, the ``start`` frame, and the server's ``ready`` ack) stays out of the
measurement, matching every other WebSocket TTS provider here. Minimax is the
closest analogue: it waits for ``task_started`` before starting its clock, as
this waits for ``ready``.

Audio is requested as raw PCM. The container formats only complete at end of
synthesis, so a first frame under ``wav`` would fold the whole synthesis into
TTFA.

Atlas is a gateway in front of a third-party synthesiser rather than a
first-party model: on the HTTP surface it returns ``x-upstream-ms`` and a
``server-timing`` breakdown attributing the bulk of each request to
``upstream``, and its error envelope is typed ``proxy_error``. The upstream is
undisclosed, so a row here may not be independent of another provider already on
the board. That is why the registry entry is early-access, off the arena, and
marked shared-inference; it belongs in the parity ledger beside the number.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

HTTP_MODELS = {"atlas-tts"}

# From GET /v1/models. All report owned_by "sabi"; none are per-account clones,
# so the benchmarked voice is the same class of object as every other provider's
# stock voice rather than something tuned for this key.
VALID_VOICES = [
    "capella",
    "dax",
    "enzo",
    "sena",
    "marlow",
    "elin",
    "corin",
    "orin",
    "isla",
    "talia",
    "serin",
    "reeve",
    "ember",
    "maitri",
]

# Pinned per provider, as everywhere else in this package. Atlas declares a rate
# on ``ready`` and ``audio.start``; that value is checked against this constant
# and a mismatch is logged rather than silently honoured, so a vendor-side change
# surfaces as a warning instead of quietly rescaling every duration and
# leading-silence offset we publish.
SAMPLE_RATE = 24000

_WS_URL = "wss://api.tts.runatlas.com/v1/audio/speech/stream"
_MAX_WS_SIZE = 16 * 1024 * 1024
_LAST_FRAMES_KEPT = 3


class AtlasTTSProvider(TTSProvider):
    """Atlas TTS provider using the streaming WebSocket API (binary PCM frames)."""

    _VALID_MODELS = frozenset(HTTP_MODELS)

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Atlas TTS model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._model = model
        self._voice = voice

        if self._voice not in VALID_VOICES:
            logger.warning("unknown_atlas_voice", voice=self._voice, fallback="dax")
            self._voice = "dax"

        api_key_secret = settings.atlas_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value():
            raise ValueError("atlas_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"atlas-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        last_frames: list[str] = []
        start: float | None = None
        first_chunk_at: float | None = None
        done = False

        try:
            async with ws_client.connect(
                _WS_URL,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                max_size=_MAX_WS_SIZE,
            ) as ws:
                await ws.send(
                    json.dumps({"type": "start", "voice": self._voice, "response_format": "pcm"})
                )

                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        if raw:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(bytes(raw))
                        continue

                    frame: dict[str, Any] = json.loads(raw)
                    frame_type = frame.get("type")

                    if frame_type == "error":
                        raise RuntimeError(f"atlas error: {frame.get('message', frame)}")

                    last_frames.append(raw)
                    del last_frames[:-_LAST_FRAMES_KEPT]

                    if frame_type == "ready":
                        self._check_declared_rate(frame, frame_type)
                        # t0 — immediately before the text submit, so connect, the
                        # start frame and this ack all stay out of TTFA.
                        start = time.monotonic()
                        await ws.send(json.dumps({"type": "text", "text": text}))
                        await ws.send(json.dumps({"type": "done"}))
                    elif frame_type == "audio.start":
                        self._check_declared_rate(frame, frame_type)
                    elif frame_type == "audio.done":
                        # audio.done carries a per-sentence error flag. Unchecked, a
                        # partial clip would be scored against the full prompt and
                        # read as a fast, fluent result.
                        if frame.get("error"):
                            index = frame.get("sentence_index")
                            raise RuntimeError(f"atlas sentence {index} failed to synthesize")
                    elif frame_type == "session.done":
                        done = True
                        break

                if not done:
                    raise RuntimeError("connection closed before the session.done frame")

        except Exception as exc:
            logger.warning("atlas_tts_error", provider="atlas", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="atlas",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                last_frames=last_frames,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="atlas",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            last_frames=last_frames,
        )

    def _check_declared_rate(self, frame: dict[str, Any], frame_type: str | None) -> None:
        """Warn when Atlas declares a rate other than the pinned one.

        The declared value is deliberately not honoured: the rate is pinned per
        provider here as it is for every other one, so a silent vendor-side change
        would otherwise rescale published durations without anyone noticing.
        """
        declared = frame.get("sample_rate")
        if isinstance(declared, bool) or not isinstance(declared, int):
            return
        if declared != SAMPLE_RATE:
            logger.warning(
                "atlas_sample_rate_mismatch",
                provider="atlas",
                model=self._model,
                voice=self._voice,
                frame=frame_type,
                declared=declared,
                pinned=SAMPLE_RATE,
            )
