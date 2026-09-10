# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The dental agent as a prompt and tool list, for models we drive ourselves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from typing import Any

from coval_bench.contracts import read_contract_file

SUITE = "dental"
PROMPT_FILE = "_source/coval-prompt.txt"
TOOLS_FILE = "tool-definitions.json"


@dataclass(frozen=True)
class DentalAgent:
    system_prompt: str
    tools: tuple[dict[str, Any], ...]


@cache
def load_dental_agent() -> DentalAgent:
    definitions = json.loads(read_contract_file(SUITE, TOOLS_FILE))
    return DentalAgent(
        system_prompt=read_contract_file(SUITE, PROMPT_FILE),
        tools=tuple({"type": "function", "function": d} for d in definitions),
    )
