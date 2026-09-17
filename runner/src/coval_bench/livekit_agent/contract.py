"""The slice of the suite contract the agent needs, read through the runner's loaders."""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from typing import Any

from coval_bench.scenarios import Stack, load_stack, public_contract_sha256, read_contract_file

TOOLS_FILE = "tool-definitions.json"
PROMPT_FILE = "system-prompt.txt"
FIRST_MESSAGE_FILE = "first-message.txt"
KEYTERMS_ENV = "DEEPGRAM_KEYTERMS"
VAD_MIN_SILENCE_SECONDS = 0.3


@dataclass(frozen=True)
class Contract:
    suite: str
    stack: Stack
    tools: list[dict[str, Any]]
    system_prompt: str
    first_message: str
    digest: str
    prompt_digest: str


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
    first_message = _text(suite, FIRST_MESSAGE_FILE, "FIRST_MESSAGE")
    return Contract(
        suite=suite,
        stack=load_stack(),
        tools=tools,
        system_prompt=system_prompt,
        first_message=first_message,
        digest=public_contract_sha256(suite),
        prompt_digest=prompt_sha256(system_prompt, first_message),
    )


def endpointing_delay(contract: Contract) -> float:
    """The fixed delay after VAD silence that lands on the pinned end-of-turn target."""
    target = contract.stack.turn_taking.end_of_turn_target_ms / 1000
    return max(target - VAD_MIN_SILENCE_SECONDS, 0.0)


def keyterms() -> list[str]:
    """The Deepgram keyterm list the stack pins for nova-3; unset is a deploy error, not silence."""
    raw = os.environ.get(KEYTERMS_ENV, "")
    terms = [term.strip() for term in raw.split(",") if term.strip()]
    if not terms:
        raise ValueError(f"{KEYTERMS_ENV} is unset; stack.json pins keyterm prompting for nova-3")
    return terms


def prompt_sha256(system_prompt: str, first_message: str) -> str:
    """Digest of the prompt and greeting actually running; overrides bypass the public digest."""
    return hashlib.sha256(f"{system_prompt}\x00{first_message}".encode()).hexdigest()


def read_tool_definitions(suite: str) -> list[dict[str, Any]]:
    import json

    tools = json.loads(read_contract_file(suite, TOOLS_FILE))
    if not isinstance(tools, list):
        raise TypeError(f"{TOOLS_FILE} must be a list")
    return tools
