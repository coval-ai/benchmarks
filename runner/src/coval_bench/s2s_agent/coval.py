# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The wire contract with Coval's WebSocket simulator.

Coval streams raw PCM both ways: binary frames in at its pipeline rate, binary
frames back at the rate its agent config declares. Text frames are JSON control
messages. Coval waits for ``session_ready`` before the persona speaks and, with
speech-activity markers on, reports when the persona starts and stops.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from typing import Any

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer

SIMULATION_HEADER = "x-coval-simulation-id"
COVAL_SEND_SAMPLE_RATE_HZ = 16000
COVAL_RECEIVE_SAMPLE_RATE_HZ = 24000
SESSION_READY = {"type": "session_ready"}
SPEECH_ACTIVITY_TYPES = frozenset({"utterance_started", "utterance_end"})


def simulation_id(headers: Mapping[str, str]) -> str | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    return lowered.get(SIMULATION_HEADER) or None


def authorized(headers: Mapping[str, str], token: str | None) -> bool:
    """A configured token must arrive as a bearer credential; none configured means open."""
    if not token:
        return True
    lowered = {key.lower(): value for key, value in headers.items()}
    scheme, _, presented = lowered.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and secrets.compare_digest(presented.strip(), token)


class CovalFrameSerializer(FrameSerializer):
    """Raw PCM each way, JSON for control; speech markers become transport messages.

    Inbound audio is resampled once here, from Coval's rate to the pipeline's,
    so the provider service downstream sees the rate it expects. Outbound audio
    reaches this serializer already at the pipeline output rate, which is what
    Coval is configured to receive, so it passes through untouched.
    """

    def __init__(
        self,
        *,
        coval_send_hz: int = COVAL_SEND_SAMPLE_RATE_HZ,
        coval_receive_hz: int = COVAL_RECEIVE_SAMPLE_RATE_HZ,
    ) -> None:
        super().__init__()
        self._coval_send_hz = coval_send_hz
        self._coval_receive_hz = coval_receive_hz
        self._pipeline_in_hz = coval_send_hz
        self._in_resampler = create_stream_resampler()
        self._out_resampler = create_stream_resampler()

    async def setup(self, setup: FrameProcessorSetup) -> None:  # type: ignore[override]
        self._pipeline_in_hz = setup.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputTransportMessageFrame | OutputTransportMessageUrgentFrame):
            return None if self.should_ignore_frame(frame) else json.dumps(frame.message)
        if isinstance(frame, AudioRawFrame):
            if frame.sample_rate == self._coval_receive_hz:
                return frame.audio
            audio: bytes = await self._out_resampler.resample(
                frame.audio, frame.sample_rate, self._coval_receive_hz
            )
            return audio or None
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            audio = data
            if self._pipeline_in_hz != self._coval_send_hz:
                audio = await self._in_resampler.resample(
                    data, self._coval_send_hz, self._pipeline_in_hz
                )
            if not audio:
                return None
            return InputAudioRawFrame(audio=audio, sample_rate=self._pipeline_in_hz, num_channels=1)
        message = _control_message(data)
        if message is None or message.get("type") not in SPEECH_ACTIVITY_TYPES:
            return None
        return InputTransportMessageFrame(message=message)


def _control_message(data: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
