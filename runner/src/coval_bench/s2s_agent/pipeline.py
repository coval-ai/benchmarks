# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""What sits between Coval's transport and the provider, in order."""

from __future__ import annotations

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


def native_processors(service: FrameProcessor) -> list[FrameProcessor]:
    """A native S2S model: the provider's own turn detection, no local VAD.

    The service only proposes turn boundaries from the provider's speech events.
    The turn processor is what turns a proposal into the interruption that drops
    audio still queued for Coval; without it the model keeps talking over the
    caller from our side even after the provider has stopped.
    """
    turns = UserTurnProcessor(user_turn_strategies=ExternalUserTurnStrategies())
    return [turns, service]
