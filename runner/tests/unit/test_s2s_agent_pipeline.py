# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("pipecat")

from pipecat.processors.frame_processor import FrameProcessor

from coval_bench.s2s_agent.pipeline import cascade_processors, context_aggregators
from coval_bench.s2s_agent.stack import load_stack


def test_cascade_runs_transcribe_accumulate_generate_synthesize() -> None:
    aggregators = context_aggregators(load_stack("bank", "cascade"))
    stt, llm, tts = FrameProcessor(), FrameProcessor(), FrameProcessor()
    assert cascade_processors(stt, llm, tts, aggregators) == [
        stt,
        aggregators.user(),
        llm,
        tts,
    ]
