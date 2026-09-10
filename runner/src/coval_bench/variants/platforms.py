# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-platform fetchers, one small table entry each.

A platform is data, not a branch. Adding one is an entry in ``FETCHERS`` plus a
function that knows three things: where its API lives, how it authenticates, and
where in its response the system prompt and tool definitions sit. Nothing else
in the benchmark knows a vendor's name.

Vapi is the first entry because it is the first variant standing, not because it
is the shape everything else must follow. Telnyx, Pipecat, LiveKit and Twilio
ConversationRelay each get an entry with the same three answers.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx

TIMEOUT_SECONDS = 30.0

# Keys whose value is a live secret. `credentialIds` is deliberately absent: it
# is an account-scoped reference that `apply` needs, and carries no key material.
SECRET_KEY = re.compile(r"(api[_-]?key|[_-]key$|secret|token|password|bearer|authorization)", re.I)


# Keys whose value names a live object in someone's account. These are not
# credentials and leak nothing on their own, but this repo is public: a dial
# target is directly abusable, and the rest map the benchmark onto real
# production objects. `display_name` and `voice_id` are deliberately absent,
# being methodology a reader needs.
IDENTIFIER_KEYS = frozenset({"id", "name", "customer_agent_id", "phone_number", "endpoint"})


def redact_identifiers(node: Any, found: list[str], path: str = "") -> Any:  # noqa: ANN401
    """Replace account-identifying values before a raw dump is written to disk."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in IDENTIFIER_KEYS and isinstance(value, str) and value:
                found.append(here)
                out[key] = "[REDACTED]"
            else:
                out[key] = redact_identifiers(value, found, here)
        return out
    if isinstance(node, list):
        return [redact_identifiers(v, found, f"{path}[{i}]") for i, v in enumerate(node)]
    return node


def redact(node: Any, found: list[str], path: str = "") -> Any:  # noqa: ANN401
    """Replace secret-looking values before a vendor config is written to disk."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if SECRET_KEY.search(key) and isinstance(value, str) and value:
                found.append(here)
                out[key] = "[REDACTED]"
            else:
                out[key] = redact(value, found, here)
        return out
    if isinstance(node, list):
        return [redact(v, found, f"{path}[{i}]") for i, v in enumerate(node)]
    return node


@dataclass(frozen=True)
class PlatformConfig:
    """What one platform's live agent looks like, normalised."""

    raw: dict[str, Any]
    system_prompt: str
    first_message: str
    tools: list[dict[str, Any]]


@dataclass(frozen=True)
class Fetcher:
    """How to reach one platform and where its fields live."""

    name: str
    api_base: str
    key_env: str
    fetch: Callable[[httpx.Client, str], PlatformConfig]
    id_help: str


def _vapi(client: httpx.Client, agent_id: str) -> PlatformConfig:
    response = client.get(f"/assistant/{agent_id}")
    response.raise_for_status()
    raw = cast("dict[str, Any]", response.json())
    model = raw.get("model") or {}
    messages = model.get("messages") or []
    system = next(
        (
            m.get("content", "")
            for m in messages
            if isinstance(m, dict) and m.get("role") == "system"
        ),
        "",
    )
    return PlatformConfig(
        raw=raw,
        system_prompt=system,
        first_message=(raw.get("firstMessage") or "").strip(),
        tools=list(model.get("tools") or []),
    )


def _telnyx(client: httpx.Client, agent_id: str) -> PlatformConfig:
    response = client.get(f"/ai/assistants/{agent_id}")
    response.raise_for_status()
    payload = cast("dict[str, Any]", response.json())
    data = payload.get("data")
    raw = data if isinstance(data, dict) else payload
    return PlatformConfig(
        raw=raw,
        system_prompt=str(raw.get("instructions") or ""),
        first_message=str(raw.get("greeting") or "").strip(),
        tools=list(raw.get("tools") or []),
    )


FETCHERS: dict[str, Fetcher] = {
    "vapi": Fetcher(
        name="vapi",
        api_base="https://api.vapi.ai",
        key_env="VAPI_API_KEY",
        fetch=_vapi,
        id_help="Vapi assistant id (uuid)",
    ),
    "telnyx": Fetcher(
        name="telnyx",
        api_base="https://api.telnyx.com/v2",
        key_env="TELNYX_API_KEY",
        fetch=_telnyx,
        id_help="Telnyx assistant id (assistant-<uuid>)",
    ),
}


def fetch_platform(platform: str, agent_id: str) -> tuple[PlatformConfig, list[str]]:
    """Fetch and redact one platform's live agent config.

    Returns the normalised config and the list of field paths that were redacted.
    """
    fetcher = FETCHERS.get(platform)
    if fetcher is None:
        known = ", ".join(sorted(FETCHERS)) or "(none)"
        raise KeyError(f"unknown platform {platform!r}; known: {known}")

    api_key = os.environ.get(fetcher.key_env)
    if not api_key:
        raise RuntimeError(f"{fetcher.key_env} is not set (needed for --platform {platform})")

    with httpx.Client(
        base_url=fetcher.api_base,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT_SECONDS,
    ) as client:
        config = fetcher.fetch(client, agent_id)

    found: list[str] = []
    redacted = redact(config.raw, found)
    return (
        PlatformConfig(
            raw=redacted,
            system_prompt=config.system_prompt,
            first_message=config.first_message,
            tools=cast("list[dict[str, Any]]", redact(config.tools, found)),
        ),
        found,
    )
