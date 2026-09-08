# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The turn contract every LLM proxied for Coval implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class TurnError(Exception):
    """A provider failed to open a session or produce a usable completion."""


@dataclass(frozen=True)
class Session:
    call_id: str
    expires_at: str | None


@dataclass(frozen=True)
class TurnResult:
    content: str
    tool_calls: tuple[dict[str, Any], ...]
    finish_reason: str
    ttft_ms: float
    total_ms: float
    output_tokens: int | None


class TurnClient(Protocol):
    async def create_session(self) -> Session: ...

    async def stream_turn(self, call_id: str, messages: list[dict[str, Any]]) -> TurnResult: ...

    async def aclose(self) -> None: ...
