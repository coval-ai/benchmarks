# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One S2S stack: its settings file, the components it names, and the prompt it runs."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from coval_bench.scenarios.annotated import AnnotatedModel

STACKS_PACKAGE = "coval_bench.s2s_agent"
STACKS_DIR = "stacks"
SCENARIOS_PACKAGE = "coval_bench.scenarios"
DEFAULT_PROMPT_FILE = "system-prompt.txt"


class ComponentPin(AnnotatedModel):
    """One role's vendor and model. ``options`` is for knobs only that vendor understands."""

    provider: str
    model: str
    options: dict[str, Any] = Field(default_factory=dict)


class SttPin(ComponentPin):
    pass


class LlmPin(ComponentPin):
    temperature: float


class TtsPin(ComponentPin):
    voice: str


class TurnTakingPin(AnnotatedModel):
    vad_stop_secs: float
    turn_analyzer: Literal["smart-turn-v3"]


class AudioPin(AnnotatedModel):
    in_sample_rate_hz: int
    out_sample_rate_hz: int


class S2SStack(AnnotatedModel):
    architecture: Literal["cascade"]
    stt: SttPin
    llm: LlmPin
    tts: TtsPin
    turn_taking: TurnTakingPin
    audio: AudioPin


@dataclass(frozen=True)
class LoadedStack:
    scenario: str
    slug: str
    stack: S2SStack
    system_prompt: str
    prompt_file: str
    digest: str


def stack_slugs() -> list[str]:
    stacks = importlib.resources.files(STACKS_PACKAGE).joinpath(STACKS_DIR)
    return sorted(
        entry.name.removesuffix(".json")
        for entry in stacks.iterdir()
        if entry.name.endswith(".json")
    )


def _stack_bytes(slug: str) -> bytes:
    path = importlib.resources.files(STACKS_PACKAGE).joinpath(STACKS_DIR, f"{slug}.json")
    if not path.is_file():
        raise ValueError(f"no stack {slug!r}; have {', '.join(stack_slugs())}")
    return path.read_bytes()


def prompt_file(scenario: str, slug: str) -> str:
    """The stack's own prompt when the scenario ships one, else the scenario default."""
    override = f"system-prompt.{slug}.txt"
    scenario_dir = importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario)
    return override if scenario_dir.joinpath(override).is_file() else DEFAULT_PROMPT_FILE


def _prompt_bytes(scenario: str, filename: str) -> bytes:
    return importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario, filename).read_bytes()


def stack_sha256(scenario: str, slug: str) -> str:
    """One SHA-256 over the stack file and the prompt actually used, in that order."""
    digest = hashlib.sha256()
    digest.update(_stack_bytes(slug))
    digest.update(_prompt_bytes(scenario, prompt_file(scenario, slug)))
    return digest.hexdigest()


def load_stack(scenario: str, slug: str) -> LoadedStack:
    stack = S2SStack.model_validate(json.loads(_stack_bytes(slug)))
    filename = prompt_file(scenario, slug)
    system_prompt = _prompt_bytes(scenario, filename).decode().strip()
    if not system_prompt:
        raise ValueError(f"{scenario}/{filename} is empty")
    return LoadedStack(scenario, slug, stack, system_prompt, filename, stack_sha256(scenario, slug))
