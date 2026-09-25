# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One S2S stack: the settings file in this package and the scenario prompt it runs."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from typing import Literal

from coval_bench.scenarios.annotated import AnnotatedModel

STACKS_PACKAGE = "coval_bench.s2s_agent"
SCENARIOS_PACKAGE = "coval_bench.scenarios"
DEFAULT_PROMPT_FILE = "system-prompt.txt"


class TurnDetectionPin(AnnotatedModel):
    type: Literal["server_vad"]
    threshold: float
    prefix_padding_ms: int
    silence_duration_ms: int


class AudioPin(AnnotatedModel):
    sample_rate_hz: int


class S2SStack(AnnotatedModel):
    provider: Literal["openai"]
    model: str
    voice: str
    reasoning_effort: str | None = None
    # None leaves endpointing on the provider's defaults.
    turn_detection: TurnDetectionPin | None = None
    audio: AudioPin


@dataclass(frozen=True)
class LoadedStack:
    scenario: str
    slug: str
    stack: S2SStack
    system_prompt: str
    prompt_file: str
    digest: str


def _stack_bytes(slug: str) -> bytes:
    return importlib.resources.files(STACKS_PACKAGE).joinpath(f"{slug}.json").read_bytes()


def prompt_file(scenario: str, slug: str) -> str:
    """The stack's own prompt when the scenario ships one, else the scenario default."""
    override = f"system-prompt.{slug}.txt"
    scenario_dir = importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario)
    return override if scenario_dir.joinpath(override).is_file() else DEFAULT_PROMPT_FILE


def _prompt_bytes(scenario: str, filename: str) -> bytes:
    return importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario, filename).read_bytes()


def stack_sha256(scenario: str, slug: str) -> str:
    """One SHA-256 over the settings file and the prompt actually used, in that order."""
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
