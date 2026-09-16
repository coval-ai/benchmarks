# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from coval_bench.config import Settings
from coval_bench.llm.benchmark import make_client
from coval_bench.llm.openai_compat import OpenAICompatClient
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import LLMConfig, RegisteredModel


@pytest.mark.parametrize(
    ("provider", "key_env", "model"),
    [("openai", "OPENAI_API_KEY", "gpt-4.1"), ("google", "GEMINI_API_KEY", "gemini-2.5-flash")],
)
def test_compat_clients_need_their_key(
    provider: str, key_env: str, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = RegisteredModel(
        benchmark=Benchmark.LLM,
        provider=provider,
        model=model,
        llm_config=LLMConfig(upstream_model=model),
        collected=True,
        published=False,
    )
    monkeypatch.delenv(key_env, raising=False)
    assert make_client(Settings(_env_file=None), registered) is None
    monkeypatch.setenv(key_env, "k")
    client = make_client(Settings(_env_file=None), registered)
    assert isinstance(client, OpenAICompatClient)
    assert model in repr(client)
