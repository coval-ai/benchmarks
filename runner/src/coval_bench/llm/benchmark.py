# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Coval run settings shared by every LLM benchmark agent."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from coval_bench.config import Settings
from coval_bench.llm.dental import load_dental_agent
from coval_bench.llm.openai_compat import OpenAICompatClient
from coval_bench.llm.phonely import PhonelyClient
from coval_bench.llm.turn import TurnClient
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import RegisteredModel

OPENAI_BASE_URL = "https://api.openai.com/v1"
# Must match the model column of the collected LLM row for this provider.
OPENAI_MODEL = "gpt-4.1"


def openai_client(settings: Settings) -> TurnClient | None:
    key = settings.openai_api_key
    if not (key and key.get_secret_value()):
        return None
    agent = load_dental_agent()
    return OpenAICompatClient(
        key.get_secret_value(),
        OPENAI_BASE_URL,
        OPENAI_MODEL,
        agent.system_prompt,
        list(agent.tools),
    )


CLIENT_FACTORIES: dict[str, Callable[[Settings], TurnClient | None]] = {
    "phonely": PhonelyClient.from_settings,
    "openai": openai_client,
}
DEFAULT_PERSONA_ID = "PN3xgmsqeLDjsNNEA2e55e"
ITERATION_COUNT = 1
TEMPLATE_MANAGED = ("agent_ids", "persona_ids", "test_set_ids", "metric_ids", "iteration_count")
_TEMPLATE_PATCH_KEYS = {
    "agent_ids": "agent_id",
    "persona_ids": "persona_id",
    "test_set_ids": "test_set_id",
}


@dataclass(frozen=True)
class ProxiedModel:
    provider: str
    model: str
    client: TurnClient


def llm_models(models: Iterable[RegisteredModel]) -> list[RegisteredModel]:
    return [model for model in models if model.benchmark is Benchmark.LLM]


def make_clients(settings: Settings) -> dict[str, TurnClient]:
    clients: dict[str, TurnClient] = {}
    for provider, factory in CLIENT_FACTORIES.items():
        client = factory(settings)
        if client is not None:
            clients[provider] = client
    return clients


def run_template_body(
    display_name: str, agent_id: str, test_set_id: str, metric_id: str
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "agent_ids": [agent_id],
        "persona_ids": [DEFAULT_PERSONA_ID],
        "test_set_ids": [test_set_id],
        "metric_ids": [metric_id],
        "iteration_count": ITERATION_COUNT,
    }


def template_patch_body(wanted: dict[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for path in paths:
        singular = _TEMPLATE_PATCH_KEYS.get(path)
        if singular is None:
            body[path] = wanted[path]
            continue
        (only,) = wanted[path]
        body[singular] = only
    return body
