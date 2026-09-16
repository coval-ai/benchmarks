"""Entrypoint: one AgentSession per inbound call on the pinned STT, LLM and TTS."""

from __future__ import annotations

import logging
import os
from typing import Any

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    vad,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

from coval_bench.livekit_agent import correlation as corr
from coval_bench.livekit_agent.contract import Contract, load_contract
from coval_bench.livekit_agent.tools import build_tools, client_from_env

logger = logging.getLogger("coval_bench.livekit_agent")

AGENT_NAME = os.environ.get("AGENT_NAME", "coval-bench-dental")
SUITE = os.environ.get("SUITE", "dental")
VAD_MIN_SILENCE_SECONDS = 0.3


def keyterms() -> list[str]:
    raw = os.environ.get("DEEPGRAM_KEYTERMS", "")
    return [term.strip() for term in raw.split(",") if term.strip()]


def endpointing_delay(contract: Contract) -> float:
    """The fixed delay after VAD silence that lands on the pinned end-of-turn target."""
    target = contract.stack.turn_taking.end_of_turn_target_ms / 1000
    return max(target - VAD_MIN_SILENCE_SECONDS, 0.0)


def build_session(contract: Contract, voice_activity: vad.VAD) -> AgentSession[Any]:
    delay = endpointing_delay(contract)
    stack = contract.stack
    return AgentSession(
        stt=deepgram.STT(model=stack.stt.model, language="en", keyterm=keyterms()),
        llm=openai.LLM(model=stack.llm.model, temperature=stack.llm.temperature),
        tts=elevenlabs.TTS(model=stack.tts.model, voice_id=stack.tts.voice_id),
        vad=voice_activity,
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing={"mode": "fixed", "min_delay": delay, "max_delay": delay},
            preemptive_generation={"enabled": False},
        ),
    )


server = AgentServer()
CONTRACT = load_contract(SUITE)
VAD = silero.VAD.load(min_silence_duration=VAD_MIN_SILENCE_SECONDS)

logger.info(
    "agent_contract",
    extra={"suite": CONTRACT.suite, "contract_digest": CONTRACT.digest, "agent_name": AGENT_NAME},
)


@server.rtc_session(agent_name=AGENT_NAME)
async def answer(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    participant = corr.sip_participant(ctx.room) or await ctx.wait_for_participant()
    correlation = corr.from_attributes(dict(participant.attributes))
    logger.info(
        "call_started",
        extra={"simulation_id": correlation.simulation_id, "source": correlation.source},
    )
    mock = client_from_env()
    tools = build_tools(CONTRACT.tools, mock.call, correlation)
    session = build_session(CONTRACT, VAD)
    await session.start(
        agent=Agent(instructions=CONTRACT.system_prompt, tools=tools), room=ctx.room
    )
    if CONTRACT.first_message:
        await session.say(CONTRACT.first_message, allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
