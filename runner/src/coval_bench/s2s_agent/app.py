# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint: one pipeline per inbound Coval call, on ``S2S_STACK`` running ``S2S_SCENARIO``."""

from __future__ import annotations

import os

import structlog
from fastapi import FastAPI, WebSocket
from pipecat.frames.frames import OutputTransportMessageUrgentFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from coval_bench.config import get_settings
from coval_bench.logging import configure_logging
from coval_bench.s2s_agent.coval import (
    SESSION_READY,
    CovalFrameSerializer,
    authorized,
    simulation_id,
)
from coval_bench.s2s_agent.services.openai import ON_SESSION_READY, RealtimeService, build_service
from coval_bench.s2s_agent.stack import LoadedStack, load_stack

logger = structlog.get_logger("coval_bench.s2s_agent")

SCENARIO = os.environ.get("S2S_SCENARIO", "bank")
STACK_SLUG = os.environ.get("S2S_STACK", "gpt-realtime")
AGENT_TOKEN = os.environ.get("S2S_AGENT_TOKEN") or None
POLICY_VIOLATION = 1008

app = FastAPI()
STACK: LoadedStack = load_stack(SCENARIO, STACK_SLUG)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {
        "status": "ok",
        "scenario": STACK.scenario,
        "stack": STACK.slug,
        "prompt": STACK.prompt_file,
        "digest": STACK.digest,
    }


@app.websocket("/ws")
async def call(websocket: WebSocket) -> None:
    if not authorized(websocket.headers, AGENT_TOKEN):
        await websocket.close(code=POLICY_VIOLATION)
        return
    await websocket.accept()
    sim = simulation_id(websocket.headers)
    log = logger.bind(
        simulation_id=sim,
        scenario=STACK.scenario,
        stack=STACK.slug,
        prompt=STACK.prompt_file,
        digest=STACK.digest,
    )
    log.info("call_started")
    await run_call(websocket, STACK, sim)
    log.info("call_ended")


async def run_call(websocket: WebSocket, loaded: LoadedStack, sim: str | None) -> None:
    api_key = get_settings().openai_api_key
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY is unset")
    transport = FastAPIWebsocketTransport(
        websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=CovalFrameSerializer(),
            allowed_origins=[],
        ),
    )
    llm = build_service(loaded, api_key.get_secret_value())
    worker = PipelineWorker(
        Pipeline([transport.input(), llm, transport.output()]),
        params=PipelineParams(
            audio_in_sample_rate=loaded.stack.audio.sample_rate_hz,
            audio_out_sample_rate=loaded.stack.audio.sample_rate_hz,
        ),
        conversation_id=sim,
        enable_rtvi=False,
    )

    @llm.event_handler(ON_SESSION_READY)  # type: ignore[untyped-decorator]
    async def _ready(_service: RealtimeService) -> None:
        # Coval holds the persona until this arrives, so a dead provider is a
        # handshake failure rather than a graded call with silence up front.
        await worker.queue_frame(OutputTransportMessageUrgentFrame(message=SESSION_READY))

    @transport.event_handler("on_client_disconnected")  # type: ignore[untyped-decorator]
    async def _gone(_transport: FastAPIWebsocketTransport, _ws: WebSocket) -> None:
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


def main() -> None:
    import uvicorn

    configure_logging()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))  # noqa: S104


if __name__ == "__main__":
    main()
