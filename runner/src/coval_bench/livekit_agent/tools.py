"""Function tools that put each call on the wire exactly as the livekit codec expects."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from livekit.agents import function_tool
from livekit.agents.llm import Tool, Toolset

from coval_bench.api.routers.mocktools import SECRET_HEADER
from coval_bench.mocktools.codecs import Correlation, codec_for
from coval_bench.platform_assets import TOOL_TIMEOUT_SECONDS

CODEC = codec_for("livekit")
Poster = Callable[[str, dict[str, Any], Correlation], Awaitable[str]]


class MockToolsClient:
    """POSTs a tool's arguments to the benchmarks API and hands the reply text to the LLM."""

    def __init__(self, base_url: str, secret: str, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._secret = secret
        self._client = client or httpx.AsyncClient(timeout=float(TOOL_TIMEOUT_SECONDS))

    async def call(self, tool: str, args: dict[str, Any], correlation: Correlation) -> str:
        request = CODEC.encode_request(tool, args, correlation)
        headers = {**request.headers, SECRET_HEADER: self._secret}
        try:
            response = await self._client.post(
                f"{self._base_url}{request.path}", json=request.body, headers=headers
            )
        except httpx.HTTPError as exc:
            return json.dumps({"error": f"tool endpoint unreachable: {type(exc).__name__}"})
        if response.status_code >= 400:
            return json.dumps({"error": f"tool endpoint returned {response.status_code}"})
        return response.text

    async def aclose(self) -> None:
        await self._client.aclose()


def build_tools(
    definitions: list[dict[str, Any]], poster: Poster, correlation: Correlation
) -> list[Tool | Toolset]:
    """One raw-schema tool per definition; name, description and parameters are verbatim."""
    tools: list[Tool | Toolset] = []
    for definition in definitions:
        name = str(definition["name"])

        async def handler(raw_arguments: dict[str, object], _name: str = name) -> str:
            return await poster(_name, dict(raw_arguments), correlation)

        schema = {
            "name": name,
            "description": str(definition["description"]),
            "parameters": definition["parameters"],
        }
        tools.append(function_tool(handler, raw_schema=schema))
    return tools


def client_from_env() -> MockToolsClient:
    return MockToolsClient(os.environ["MOCK_TOOLS_BASE_URL"], os.environ["MOCK_TOOLS_SECRET"])
