# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reading a platform's live agent: secret redaction and the Retell lookup."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# Keys whose value is a live secret. `credentialIds` is deliberately absent: it
# is an account-scoped reference that `apply` needs, and carries no key material.
SECRET_KEY = re.compile(r"(api[_-]?key|[_-]key$|secret|token|password|bearer|authorization)", re.I)


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


RETELL_ENGINE = "retell-llm"
Getter = Callable[[str, dict[str, Any] | None], dict[str, Any]]


def retell_engine(agent: dict[str, Any]) -> tuple[str, int | None]:
    """The Retell LLM an agent answers with; prompt, greeting and tools live there."""
    engine = agent.get("response_engine") or {}
    if engine.get("type") != RETELL_ENGINE:
        raise RuntimeError(
            f"retell agent {agent.get('agent_id')!r} uses response engine "
            f"{engine.get('type')!r}; only {RETELL_ENGINE} carries the prompt and tools"
        )
    version = engine.get("version")
    return str(engine["llm_id"]), int(version) if version is not None else None


def read_retell(get: Getter, agent_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The agent and the single-prompt Retell LLM it answers with, at the version it pins."""
    agent = get(f"/get-agent/{agent_id}", None)
    llm_id, version = retell_engine(agent)
    llm = get(f"/get-retell-llm/{llm_id}", {"version": version} if version is not None else None)
    states = llm.get("states") or []
    if states:
        raise RuntimeError(
            f"retell llm {llm_id!r} has {len(states)} states; their prompts and tools stay live "
            "outside the contract, so only a single-prompt LLM is supported"
        )
    return agent, llm
