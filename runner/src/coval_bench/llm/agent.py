# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The agent under test as a prompt and tool list, read from a suite contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from typing import Any

from coval_bench.contracts import read_contract_file

SYSTEM_PROMPT_FILE = "system-prompt.txt"
TOOLS_FILE = "tool-definitions.json"


@dataclass(frozen=True)
class Agent:
    system_prompt: str
    tools: tuple[dict[str, Any], ...]


@cache
def load_agent(suite: str) -> Agent:
    definitions = json.loads(read_contract_file(suite, TOOLS_FILE))
    return Agent(
        system_prompt=read_contract_file(suite, SYSTEM_PROMPT_FILE),
        tools=tuple({"type": "function", "function": d} for d in definitions),
    )
