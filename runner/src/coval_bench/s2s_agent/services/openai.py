# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI's realtime model through Pipecat's stock service, settings from the stack file."""

from __future__ import annotations

from typing import Any

from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from coval_bench.s2s_agent.stack import LoadedStack

ON_SESSION_READY = "on_session_ready"


class RealtimeService(OpenAIRealtimeLLMService):
    """The stock service plus one event: the provider accepted our session settings."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self._session_ready_announced = False
        self._register_event_handler(ON_SESSION_READY)

    async def _handle_evt_session_updated(self, evt: Any) -> None:  # noqa: ANN401
        await super()._handle_evt_session_updated(evt)  # type: ignore[no-untyped-call]
        if not self._session_ready_announced:
            self._session_ready_announced = True
            await self._call_event_handler(ON_SESSION_READY)


def session_properties(loaded: LoadedStack) -> events.SessionProperties:
    stack = loaded.stack
    turn_detection: events.TurnDetection | None = None
    if stack.turn_detection is not None:
        turn_detection = events.TurnDetection(
            type=stack.turn_detection.type,
            threshold=stack.turn_detection.threshold,
            prefix_padding_ms=stack.turn_detection.prefix_padding_ms,
            silence_duration_ms=stack.turn_detection.silence_duration_ms,
        )
    reasoning = events.Reasoning(effort=stack.reasoning_effort) if stack.reasoning_effort else None
    return events.SessionProperties(
        output_modalities=["audio"],
        audio=events.AudioConfiguration(
            input=events.AudioInput(format=events.PCMAudioFormat(), turn_detection=turn_detection),
            output=events.AudioOutput(format=events.PCMAudioFormat(), voice=stack.voice),
        ),
        reasoning=reasoning,
    )


def build_service(loaded: LoadedStack, api_key: str) -> RealtimeService:
    return RealtimeService(
        api_key=api_key,
        settings=RealtimeService.Settings(
            model=loaded.stack.model,
            system_instruction=loaded.system_prompt,
            session_properties=session_properties(loaded),
        ),
    )
