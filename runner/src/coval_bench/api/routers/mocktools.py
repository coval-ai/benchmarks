# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The mock tool endpoint the benchmarked agents call.

``POST /mock/{tool}`` with the tool's arguments as a JSON object. The response
is whichever seed the resolver picks, held to a fixed latency budget so that a
tool call costs the same on every platform under test. Without that, a variant
would be measured partly on how fast our mock happened to answer it.

Mounted outside ``/v1``: this is not part of the public read API, it is an
appliance the agents talk to. It carries a shared secret rather than Clerk —
the caller is a voice platform's tool runner, not a browser — and it is not rate
limited, because a burst of calls is a scenario doing its job and a 429
mid-conversation would be graded as the agent breaking.
"""

from __future__ import annotations

import asyncio
import hmac
import time
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Body, Depends, Header, HTTPException
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request
from starlette.responses import JSONResponse

from coval_bench.api.deps import get_pool, get_settings
from coval_bench.config import Settings
from coval_bench.db.mock_tool_store import record_call
from coval_bench.mocktools.dispatch import Dispatcher

logger = structlog.get_logger("coval_bench.mocktools")

router = APIRouter(prefix="/mock", tags=["mock-tools"])

SIMULATION_HEADER = "X-Coval-Simulation-Id"
CALLER_HEADER = "X-Coval-Caller-Number"
SECRET_HEADER = "X-Mock-Tools-Key"  # noqa: S105 — a header name, not a credential


def require_mock_secret(
    x_mock_tools_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Gate the appliance behind its shared secret.

    Fails closed when no secret is configured: an open mock would let anyone
    write rows into the tool-call log, and a poisoned log is indistinguishable
    from an agent that behaved badly.
    """
    expected = settings.mock_tools_secret
    if expected is None:
        raise HTTPException(503, "mock tools are not configured")
    if x_mock_tools_key is None or not hmac.compare_digest(
        x_mock_tools_key.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    ):
        raise HTTPException(401, f"a valid {SECRET_HEADER} is required")


def get_dispatcher(request: Request) -> Dispatcher:
    """The dispatcher the lifespan built, or 503.

    Deliberately does not build one here. Loading the fixtures can mean a call
    out to object storage, and a service that failed to load them at startup will
    not succeed on request four hundred — retrying per request would put a round
    trip in front of every tool call, in the middle of a live conversation, and a
    miss is not something a cache prevents. Startup has already logged why.
    """
    dispatcher: Dispatcher | None = getattr(request.app.state, "mock_dispatcher", None)
    if dispatcher is None:
        raise HTTPException(503, "mock tool fixtures are not loaded")
    return dispatcher


@router.post("/{tool}", dependencies=[Depends(require_mock_secret)])
async def call_tool(
    tool: str,
    background: BackgroundTasks,
    args: dict[str, Any] = Body(default=None),
    x_coval_simulation_id: str | None = Header(default=None),
    x_coval_caller_number: str | None = Header(default=None),
    dispatcher: Dispatcher = Depends(get_dispatcher),
    settings: Settings = Depends(get_settings),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> JSONResponse:
    """Answer one tool call, hold it to the latency budget, and log it."""
    started = time.perf_counter()
    call_args = args if args is not None else {}
    outcome = dispatcher.call(tool, call_args)

    # Hold every answer to the same budget so tool time is a constant across
    # variants rather than a property of how much work this particular seed took.
    budget_s = settings.mock_tools_latency_ms / 1000
    remaining = budget_s - (time.perf_counter() - started)
    if remaining > 0:
        await asyncio.sleep(remaining)
    latency_ms = (time.perf_counter() - started) * 1000

    # Logged after the response goes out: the row is telemetry and must not be
    # charged to the latency the platform observes.
    background.add_task(
        record_call,
        pool,
        tool=tool,
        args=call_args,
        response=outcome.response,
        latency_ms=latency_ms,
        matched_seed=outcome.resolution.matched_seed if outcome.resolution else None,
        simulation_id=x_coval_simulation_id,
        caller_number=x_coval_caller_number,
    )
    logger.info(
        "mock_tool_call",
        tool=tool,
        status=outcome.http_status,
        mode=outcome.resolution.mode if outcome.resolution else "rejected",
        seed=outcome.resolution.matched_seed if outcome.resolution else None,
        simulation_id=x_coval_simulation_id,
    )
    return JSONResponse(outcome.response, status_code=outcome.http_status)
