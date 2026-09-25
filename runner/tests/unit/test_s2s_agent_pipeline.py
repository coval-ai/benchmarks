import asyncio
import time

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    Frame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    ProposedUserStartedSpeakingFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from coval_bench.s2s_agent.pipeline import native_processors

SAMPLE_RATE = 24000
CHUNK = bytes(2 * SAMPLE_RATE * 20 // 1000)
CHUNKS_QUEUED = 100  # two seconds of bot audio


class ProviderStub(FrameProcessor):
    """Streams a reply on "speak"; proposes a caller turn on "caller", like the realtime service."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame):
            if frame.message["type"] == "speak":
                for _ in range(CHUNKS_QUEUED):
                    await self.push_frame(
                        TTSAudioRawFrame(audio=CHUNK, sample_rate=SAMPLE_RATE, num_channels=1)
                    )
            elif frame.message["type"] == "caller":
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
            return
        await self.push_frame(frame, direction)


class PacedRecorder(BaseOutputTransport):
    """An output transport that plays audio in real time and records what got out."""

    def __init__(self) -> None:
        super().__init__(TransportParams(audio_out_enabled=True, audio_out_sample_rate=SAMPLE_RATE))
        self.written: list[float] = []
        self.interrupted_at: float | None = None

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InterruptionFrame) and self.interrupted_at is None:
            self.interrupted_at = time.monotonic()
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self.written.append(time.monotonic())
        await asyncio.sleep(len(frame.audio) / (2 * SAMPLE_RATE))
        return True


@pytest.mark.asyncio
async def test_a_provider_speech_start_drops_the_audio_still_queued_for_coval() -> None:
    recorder = PacedRecorder()
    await run_test(
        Pipeline([*native_processors(ProviderStub()), recorder]),
        frames_to_send=[
            InputTransportMessageFrame(message={"type": "speak"}),
            SleepFrame(0.2),
            InputTransportMessageFrame(message={"type": "caller"}),
            SleepFrame(0.5),
        ],
        expected_down_frames=None,
    )
    assert recorder.interrupted_at is not None
    assert 0 < len(recorder.written) < CHUNKS_QUEUED // 2
    # Queued playback would keep writing every 20 ms; after the interruption the
    # only write left is the transport's end-of-stream flush, half a second on.
    quiet = (recorder.interrupted_at + 0.05, recorder.interrupted_at + 0.4)
    assert not [at for at in recorder.written if quiet[0] < at < quiet[1]]
