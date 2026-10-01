# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One S2S stack: its settings file here, the pinned components, and the prompt it runs."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from typing import Literal

from coval_bench import scenarios
from coval_bench.scenarios.annotated import AnnotatedModel

STACKS_PACKAGE = "coval_bench.s2s_agent"
SCENARIOS_PACKAGE = "coval_bench.scenarios"
COMPONENTS_FILE = "stack.json"
DEFAULT_PROMPT_FILE = "system-prompt.txt"


class TurnTakingPin(AnnotatedModel):
    vad_stop_secs: float
    turn_analyzer: Literal["smart-turn-v3"]


class AudioPin(AnnotatedModel):
    in_sample_rate_hz: int
    out_sample_rate_hz: int


class S2SStack(AnnotatedModel):
    architecture: Literal["cascade"]
    # A cascade never names its own models; it points at the pinned layer.
    components: Literal["scenarios/stack.json"]
    turn_taking: TurnTakingPin
    audio: AudioPin


@dataclass(frozen=True)
class LoadedStack:
    scenario: str
    slug: str
    stack: S2SStack
    components: scenarios.Stack
    system_prompt: str
    prompt_file: str
    digest: str


def _stack_bytes(slug: str) -> bytes:
    return importlib.resources.files(STACKS_PACKAGE).joinpath(f"{slug}.json").read_bytes()


def _components_bytes() -> bytes:
    return importlib.resources.files(SCENARIOS_PACKAGE).joinpath(COMPONENTS_FILE).read_bytes()


def prompt_file(scenario: str, slug: str) -> str:
    """The stack's own prompt when the scenario ships one, else the scenario default."""
    override = f"system-prompt.{slug}.txt"
    scenario_dir = importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario)
    return override if scenario_dir.joinpath(override).is_file() else DEFAULT_PROMPT_FILE


def _prompt_bytes(scenario: str, filename: str) -> bytes:
    return importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario, filename).read_bytes()


def stack_sha256(scenario: str, slug: str) -> str:
    """One SHA-256 over the stack file, the pinned components and the prompt, in that order."""
    digest = hashlib.sha256()
    digest.update(_stack_bytes(slug))
    digest.update(_components_bytes())
    digest.update(_prompt_bytes(scenario, prompt_file(scenario, slug)))
    return digest.hexdigest()


def load_stack(scenario: str, slug: str) -> LoadedStack:
    stack = S2SStack.model_validate(json.loads(_stack_bytes(slug)))
    filename = prompt_file(scenario, slug)
    system_prompt = _prompt_bytes(scenario, filename).decode().strip()
    if not system_prompt:
        raise ValueError(f"{scenario}/{filename} is empty")
    return LoadedStack(
        scenario=scenario,
        slug=slug,
        stack=stack,
        components=scenarios.load_stack(),
        system_prompt=system_prompt,
        prompt_file=filename,
        digest=stack_sha256(scenario, slug),
    )
