# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The mock tool endpoint the benchmarked agents call.

``POST /mock/{platform}/{tool}`` for platforms that name the tool in the URL and
``POST /mock/{platform}`` for those that name it in the body. The platform's codec
unwraps the request into tool calls, the dispatcher answers each from the seeded
fixtures, and the codec wraps the answers back into that platform's reply shape.
Nothing past the codec knows which platform asked.

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
from coval_bench.mocktools.codecs import Codec, codec_for
from coval_bench.mocktools.dispatch import Dispatcher

logger = structlog.get_logger("coval_bench.mocktools")

router = APIRouter(prefix="/mock", tags=["mock-tools"])

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


def get_codec(platform: str) -> Codec:
    try:
        return codec_for(platform)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


async def _answer(
    codec: Codec,
    tool: str | None,
    body: dict[str, Any] | None,
    request: Request,
    background: BackgroundTasks,
    dispatcher: Dispatcher,
    settings: Settings,
    pool: AsyncConnectionPool[Any],
) -> JSONResponse:
    started = time.perf_counter()
    calls = codec.decode(body, request.headers, tool)
    correlation = codec.correlate(body, request.headers)
    outcomes = [dispatcher.call(call.tool, call.args) for call in calls]

    budget_s = len(calls) * settings.mock_tools_latency_ms / 1000
    remaining = budget_s - (time.perf_counter() - started)
    if remaining > 0:
        await asyncio.sleep(remaining)
    latency_ms = (time.perf_counter() - started) * 1000 / max(len(calls), 1)

    for call, outcome in zip(calls, outcomes, strict=True):
        background.add_task(
            record_call,
            pool,
            tool=call.tool,
            args=call.args,
            response=outcome.response,
            latency_ms=latency_ms,
            matched_seed=outcome.resolution.matched_seed if outcome.resolution else None,
            simulation_id=correlation.simulation_id,
            caller_number=correlation.caller_number,
        )
        logger.info(
            "mock_tool_call",
            platform=codec.name,
            tool=call.tool,
            status=outcome.http_status,
            mode=outcome.resolution.mode if outcome.resolution else "rejected",
            seed=outcome.resolution.matched_seed if outcome.resolution else None,
            simulation_id=correlation.simulation_id,
            correlation_source=correlation.source,
        )
    if not calls:
        logger.warning("mock_tool_call_empty", platform=codec.name)

    payload, status = codec.encode(calls, outcomes)
    return JSONResponse(payload, status_code=status)


@router.post("/{platform}", dependencies=[Depends(require_mock_secret)])
async def call_tools(
    platform: str,
    request: Request,
    background: BackgroundTasks,
    body: dict[str, Any] | None = Body(default=None),
    dispatcher: Dispatcher = Depends(get_dispatcher),
    settings: Settings = Depends(get_settings),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> JSONResponse:
    codec = get_codec(platform)
    if codec.tool_in_path:
        raise HTTPException(
            404, f"{platform} names the tool in the path: /mock/{platform}/{{tool}}"
        )
    return await _answer(codec, None, body, request, background, dispatcher, settings, pool)


@router.post("/{platform}/{tool}", dependencies=[Depends(require_mock_secret)])
async def call_tool(
    platform: str,
    tool: str,
    request: Request,
    background: BackgroundTasks,
    body: dict[str, Any] | None = Body(default=None),
    dispatcher: Dispatcher = Depends(get_dispatcher),
    settings: Settings = Depends(get_settings),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> JSONResponse:
    codec = get_codec(platform)
    if not codec.tool_in_path:
        raise HTTPException(404, f"{platform} names the tool in the body: /mock/{platform}")
    return await _answer(codec, tool, body, request, background, dispatcher, settings, pool)
