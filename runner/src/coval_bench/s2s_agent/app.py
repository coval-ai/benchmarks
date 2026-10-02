# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Talk to a stack on this machine's microphone and speakers. Ctrl-C hangs up."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Callable

from pipecat.frames.frames import Frame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

from coval_bench.config import get_settings
from coval_bench.logging import configure_logging
from coval_bench.s2s_agent.pipeline import cascade_processors, context_aggregators
from coval_bench.s2s_agent.services import resolve
from coval_bench.s2s_agent.stack import LoadedStack, load_stack, stack_slugs


async def run(loaded: LoadedStack) -> None:
    # pyaudio comes with the s2s-agent-local extra only; nothing else needs it.
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

    services = resolve(loaded, get_settings())
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
    middle = cascade_processors(services.stt, services.llm, services.tts, aggregators)
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


def choose_stack(scenario: str, read: Callable[[str], str] = input) -> LoadedStack:
    """Print every stack's STT, LLM and TTS and take a number."""
    stacks = [load_stack(scenario, slug) for slug in stack_slugs()]
    width = max(len(loaded.slug) for loaded in stacks)
    for index, loaded in enumerate(stacks, start=1):
        pins = loaded.stack
        print(
            f"{index}. {loaded.slug:<{width}}  "
            f"stt {pins.stt.provider}/{pins.stt.model}  "
            f"llm {pins.llm.provider}/{pins.llm.model}  "
            f"tts {pins.tts.provider}/{pins.tts.model}"
        )
    while True:
        answer = read(f"Stack [1-{len(stacks)}]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(stacks):
            return stacks[int(answer) - 1]
        print(f"pick a number from 1 to {len(stacks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default=os.environ.get("S2S_STACK", "cascade-nova3-gpt41-flash"))
    parser.add_argument("--scenario", default=os.environ.get("S2S_SCENARIO", "bank"))
    parser.add_argument("--tui", action="store_true", help="pick the stack from a list instead")
    args = parser.parse_args()
    loaded = choose_stack(args.scenario) if args.tui else load_stack(args.scenario, args.stack)
    configure_logging()
    asyncio.run(run(loaded))


if __name__ == "__main__":
    main()
