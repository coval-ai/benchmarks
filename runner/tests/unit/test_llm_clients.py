# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from coval_bench.config import Settings
from coval_bench.llm.benchmark import make_clients
from coval_bench.llm.openai_compat import OpenAICompatClient


@pytest.mark.parametrize(
    ("provider", "key_env", "model"),
    [("openai", "OPENAI_API_KEY", "gpt-4.1")],
)
def test_compat_clients_need_their_key(
    provider: str, key_env: str, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(key_env, raising=False)
    assert provider not in make_clients(Settings(_env_file=None))
    monkeypatch.setenv(key_env, "k")
    clients = make_clients(Settings(_env_file=None))
    assert isinstance(clients[provider], OpenAICompatClient)
    assert model in repr(clients[provider])
