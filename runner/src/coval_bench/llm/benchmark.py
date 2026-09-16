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
from coval_bench.registries.models import RegisteredModel

# Preserve existing provider-only routes and Coval identities during rollout.
LEGACY_MODELS = {
    "phonely": "phonely-agent",
    "openai": "gpt-4.1",
    "google": "gemini-2.5-flash",
}
# These registry names are distinct benchmark configurations of upstream gpt-5.
GPT5_REASONING = {"gpt-5-minimal": "minimal", "gpt-5-medium": "medium"}
ModelKey = tuple[str, str]


def make_client(settings: Settings, provider: str, model: str) -> TurnClient | None:
    if provider == "phonely":
        # Phonely hosts one configured agent rather than accepting a model name.
        return PhonelyClient.from_settings(settings) if model == "phonely-agent" else None
    if provider == "openai":
        key = settings.openai_api_key
        base_url = "https://api.openai.com/v1"
        effort = GPT5_REASONING.get(model)
        upstream_model = "gpt-5" if effort else model
        extra_body = {"reasoning_effort": effort} if effort else {}
    elif provider == "google":
        key = settings.gemini_api_key
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        upstream_model = model
        extra_body = {"reasoning_effort": "none"}
    else:
        return None
    if not (key and key.get_secret_value()):
        return None
    agent = load_agent(scenarios.ACTIVE.contract)
    return OpenAICompatClient(
        key.get_secret_value(),
        base_url,
        upstream_model,
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


def make_clients(settings: Settings) -> dict[ModelKey, TurnClient]:
    clients: dict[ModelKey, TurnClient] = {}
    for provider, model in LEGACY_MODELS.items():
        client = make_client(settings, provider, model)
        if client is not None:
            clients[provider, model] = client
    return clients


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
