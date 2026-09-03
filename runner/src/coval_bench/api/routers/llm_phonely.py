# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticated OpenAI-compatible proxy for Phonely text-agent turns."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from coval_bench.api.deps import (
    bearer_token,
    get_phonely_client,
    get_pool,
    get_settings,
    secret_matches,
)
from coval_bench.config import Settings
from coval_bench.db.llm_turns import insert_turn
from coval_bench.llm.phonely import PhonelyClient, PhonelyError, TurnResult

logger = structlog.get_logger("coval_bench.api.llm_phonely")

router = APIRouter(prefix="/llm/phonely", tags=["llm-phonely"])

_MESSAGE_KEYS = frozenset({"role", "content", "name", "tool_calls", "tool_call_id"})
_TURN_TIMEOUT_S = 150.0


# Only model (the Phonely callId) and messages reach Phonely; its agent owns tools and
# sampling, so other OpenAI request fields are dropped.
class ChatRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[dict[str, Any]]
    simulation_id: str | None = None
    stream: bool | None = None


def require_proxy_secret(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Authenticate Coval with the proxy's shared bearer secret."""
    if settings.llm_proxy_secret is None:
        raise HTTPException(503, "LLM proxy is not configured")
    if not secret_matches(bearer_token(authorization), settings.llm_proxy_secret):
        raise HTTPException(
            401,
            "a valid proxy bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in message.items() if key in _MESSAGE_KEYS}
        for message in messages
    ]


async def _record_turn(
    pool: AsyncConnectionPool[Any],
    *,
    simulation_id: str,
    turn_index: int,
    result: TurnResult,
) -> None:
    try:
        await insert_turn(
            pool,
            simulation_id=simulation_id,
            turn_index=turn_index,
            provider="phonely",
            model="phonely-agent",
            ttft_ms=result.ttft_ms,
            total_ms=result.total_ms,
            output_tokens=result.output_tokens,
        )
    except Exception:
        logger.error(
            "llm_turn_not_recorded",
            simulation_id=simulation_id,
            turn_index=turn_index,
            exc_info=True,
        )
    else:
        logger.info("llm_turn_recorded", simulation_id=simulation_id, turn_index=turn_index)


def _completion(call_id: str, result: TurnResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        message["tool_calls"] = list(result.tool_calls)
    output_tokens = result.output_tokens or 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": call_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": output_tokens,
            "total_tokens": output_tokens,
        },
    }


@router.post("/session", dependencies=[Depends(require_proxy_secret)])
async def create_session(
    client: PhonelyClient = Depends(get_phonely_client),
) -> dict[str, str | None]:
    try:
        session = await client.create_session()
    except PhonelyError as exc:
        logger.warning("phonely_session_failed", error=str(exc))
        raise HTTPException(502, "Phonely session creation failed") from exc
    return {"sessionId": session.call_id, "expiresAt": session.expires_at}


@router.post("/chat", dependencies=[Depends(require_proxy_secret)])
async def chat(
    body: ChatRequest,
    background: BackgroundTasks,
    client: PhonelyClient = Depends(get_phonely_client),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> JSONResponse:
    if body.stream:
        raise HTTPException(400, "streaming responses are not supported")
    messages = _messages(body.messages)
    turn_index = sum(message.get("role") == "assistant" for message in messages)
    try:
        async with asyncio.timeout(_TURN_TIMEOUT_S):
            result = await client.stream_turn(body.model, messages)
    except TimeoutError as exc:
        logger.warning("phonely_turn_timed_out", turn_index=turn_index)
        raise HTTPException(504, "Phonely completion timed out") from exc
    except PhonelyError as exc:
        logger.warning("phonely_turn_failed", turn_index=turn_index, error=str(exc))
        raise HTTPException(502, "Phonely completion failed") from exc

    if body.simulation_id:
        background.add_task(
            _record_turn,
            pool,
            simulation_id=body.simulation_id,
            turn_index=turn_index,
            result=result,
        )
    else:
        logger.warning("llm_turn_unattributed", turn_index=turn_index)
    return JSONResponse(
        _completion(body.model, result),
        headers={
            "X-Coval-Ttft-Ms": f"{result.ttft_ms:.3f}",
            "X-Coval-Total-Ms": f"{result.total_ms:.3f}",
        },
    )
