# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Coval run settings shared by every LLM benchmark agent."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from coval_bench import scenarios
from coval_bench.config import Settings
from coval_bench.llm.agent import load_agent
from coval_bench.llm.openai_compat import OpenAICompatClient
from coval_bench.llm.phonely import PhonelyClient
from coval_bench.llm.turn import TurnClient
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import LLMConfig, RegisteredModel

ModelKey = tuple[str, str]
ClientKey = tuple[str, str, LLMConfig]


def make_client(settings: Settings, model: RegisteredModel) -> TurnClient | None:
    config = model.llm_config
    if config is None:
        return None
    if model.provider == "phonely":
        return PhonelyClient.from_settings(settings)
    if model.provider == "openai":
        key = settings.openai_api_key
        base_url = "https://api.openai.com/v1"
    elif model.provider == "google":
        key = settings.gemini_api_key
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    else:
        return None
    extra_body = {"reasoning_effort": config.reasoning_effort} if config.reasoning_effort else {}
    if not (key and key.get_secret_value()):
        return None
    agent = load_agent(scenarios.ACTIVE.contract)
    return OpenAICompatClient(
        key.get_secret_value(),
        base_url,
        config.upstream_model,
        agent.system_prompt,
        list(agent.tools),
        extra_body=extra_body,
    )


ITERATION_COUNT = 1
TEMPLATE_MANAGED = ("agent_ids", "persona_ids", "test_set_ids", "metric_ids", "iteration_count")


@dataclass(frozen=True)
class ProxiedModel:
    provider: str
    model: str
    client: TurnClient


def llm_models(models: Iterable[RegisteredModel]) -> list[RegisteredModel]:
    return [model for model in models if model.benchmark is Benchmark.LLM]


def run_template_body(
    display_name: str, agent_id: str, test_set_id: str, metric_id: str, persona_id: str
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "agent_ids": [agent_id],
        "persona_ids": [persona_id],
        "test_set_ids": [test_set_id],
        "metric_ids": [metric_id],
        "iteration_count": ITERATION_COUNT,
    }


def template_patch_body(wanted: dict[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    return {path: wanted[path] for path in paths}
