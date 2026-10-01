# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Talk to the cascade on this machine's microphone and speakers. Ctrl-C hangs up."""

from __future__ import annotations

import argparse
import asyncio
import os

from pipecat.frames.frames import Frame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from coval_bench.config import get_settings
from coval_bench.logging import configure_logging
from coval_bench.s2s_agent.pipeline import cascade_processors, context_aggregators
from coval_bench.s2s_agent.services.cascade import Keys, build_llm, build_stt, build_tts
from coval_bench.s2s_agent.stack import LoadedStack, load_stack


async def run(loaded: LoadedStack) -> None:
    keys = Keys.from_settings(get_settings())
    audio = loaded.stack.audio
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=audio.in_sample_rate_hz,
            audio_out_sample_rate=audio.out_sample_rate_hz,
        )
    )
    aggregators = context_aggregators(loaded)
    middle = cascade_processors(
        build_stt(loaded, keys), build_llm(loaded, keys), build_tts(loaded, keys), aggregators
    )
    worker = PipelineWorker(
        Pipeline([transport.input(), *middle, transport.output(), aggregators.assistant()]),
        params=PipelineParams(
            audio_in_sample_rate=audio.in_sample_rate_hz,
            audio_out_sample_rate=audio.out_sample_rate_hz,
        ),
        enable_rtvi=False,
    )

    @worker.event_handler("on_pipeline_started")  # type: ignore[untyped-decorator]
    async def _greet(started: PipelineWorker, _frame: Frame) -> None:
        # Fires once StartFrame has passed every processor, so every service is up.
        await started.queue_frame(LLMRunFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default=os.environ.get("S2S_STACK", "cascade"))
    parser.add_argument("--scenario", default=os.environ.get("S2S_SCENARIO", "bank"))
    args = parser.parse_args()
    configure_logging()
    asyncio.run(run(load_stack(args.scenario, args.stack)))


if __name__ == "__main__":
    main()
