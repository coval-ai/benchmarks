# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""What sits between the transport's input and output, in order."""

from __future__ import annotations

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from coval_bench.s2s_agent.stack import LoadedStack


def context_aggregators(loaded: LoadedStack) -> LLMContextAggregatorPair:
    """The conversation context and the turn-taking the stack pins around it.

    Silero decides when the caller is speaking; smart-turn decides when a pause
    is the end of the turn rather than a breath. Both are local models shipped
    with Pipecat, so the cascade's endpointing is the same on every machine.
    """
    turn_taking = loaded.stack.turn_taking
    context = LLMContext([{"role": "system", "content": loaded.system_prompt}])
    return LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=turn_taking.vad_stop_secs)),
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
        ),
    )


def cascade_processors(
    stt: FrameProcessor,
    llm: FrameProcessor,
    tts: FrameProcessor,
    aggregators: LLMContextAggregatorPair,
) -> list[FrameProcessor]:
    """Between transport input and output: transcribe, accumulate, generate, synthesize.

    The assistant-side aggregator is not here; it belongs after the transport's
    output so it records what was actually said, and the caller places it.
    """
    return [stt, aggregators.user(), llm, tts]
