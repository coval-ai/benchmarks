"""The slice of the suite contract the agent needs, read through the runner's loaders."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

from coval_bench.scenarios import Stack, load_stack, public_contract_sha256, read_contract_file

TOOLS_FILE = "tool-definitions.json"
PROMPT_FILE = "system-prompt.txt"
FIRST_MESSAGE_FILE = "first-message.txt"


@dataclass(frozen=True)
class Contract:
    suite: str
    stack: Stack
    tools: list[dict[str, Any]]
    system_prompt: str
    first_message: str
    digest: str


def _text(suite: str, filename: str, env: str) -> str:
    """The suite file, unless the plain or base64 environment override is set."""
    encoded = os.environ.get(f"{env}_B64")
    if encoded:
        return base64.b64decode(encoded).decode("utf-8").strip()
    override = os.environ.get(env)
    if override:
        return override
    try:
        return read_contract_file(suite, filename).strip()
    except FileNotFoundError:
        return ""


def load_contract(suite: str) -> Contract:
    tools = read_tool_definitions(suite)
    system_prompt = _text(suite, PROMPT_FILE, "SYSTEM_PROMPT")
    if not system_prompt:
        raise ValueError(f"no system prompt: commit {suite}/{PROMPT_FILE} or set SYSTEM_PROMPT")
    return Contract(
        suite=suite,
        stack=load_stack(),
        tools=tools,
        system_prompt=system_prompt,
        first_message=_text(suite, FIRST_MESSAGE_FILE, "FIRST_MESSAGE"),
        digest=public_contract_sha256(suite),
    )


def read_tool_definitions(suite: str) -> list[dict[str, Any]]:
    import json

    tools = json.loads(read_contract_file(suite, TOOLS_FILE))
    if not isinstance(tools, list):
        raise TypeError(f"{TOOLS_FILE} must be a list")
    return tools
