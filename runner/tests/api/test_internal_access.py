# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the early-access embargo (``X-Internal-Key``).

EARLY_ACCESS models must never appear in public responses of the data
endpoints; a request presenting the internal key sees them everywhere.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from coval_bench.api.internal import is_internal
from coval_bench.config import Settings
from coval_bench.registries import MODEL_REGISTRY, Benchmark, ModelStatus, RegisteredModel
from tests.api.conftest import (
    INTERNAL_API_KEY,
    _fill_buckets,
    _insert_result,
    _insert_run,
    _refresh_mv,
)

_INTERNAL_HEADERS = {"X-Internal-Key": INTERNAL_API_KEY}
_WRONG_HEADERS = {"X-Internal-Key": "not-the-key"}

_EA_PROVIDER = "acme"
_EA_MODEL = "unreleased-stt"


@pytest.fixture
def early_access_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extend the registry with one EARLY_ACCESS STT model for the duration of a test."""
    patched = [
        *MODEL_REGISTRY,
        RegisteredModel(
            benchmark=Benchmark.STT,
            provider=_EA_PROVIDER,
            model=_EA_MODEL,
            status=ModelStatus.EARLY_ACCESS,
        ),
    ]
    monkeypatch.setattr("coval_bench.api.internal.MODEL_REGISTRY", patched)
    monkeypatch.setattr("coval_bench.api.routers.providers.MODEL_REGISTRY", patched)


async def _seed_ea_and_public_rows(postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, provider=_EA_PROVIDER, model=_EA_MODEL)
    await _insert_result(postgresql, run_id, provider="deepgram", model="nova-3")


def _models_in(results: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(r["provider"], r["model"]) for r in results}


@pytest.mark.usefixtures("early_access_registry")
async def test_results_hides_early_access_from_public(client: AsyncClient, postgresql: Any) -> None:
    """Public and wrong-key callers never see EARLY_ACCESS rows on /v1/results."""
    await _seed_ea_and_public_rows(postgresql)

    for headers in ({}, _WRONG_HEADERS):
        response = await client.get("/v1/results", headers=headers)
        assert response.status_code == 200
        models = _models_in(response.json()["results"])
        assert ("deepgram", "nova-3") in models
        assert (_EA_PROVIDER, _EA_MODEL) not in models


@pytest.mark.usefixtures("early_access_registry")
async def test_results_serves_early_access_to_internal(
    client: AsyncClient, postgresql: Any
) -> None:
    """The internal key unlocks EARLY_ACCESS rows on /v1/results."""
    await _seed_ea_and_public_rows(postgresql)

    response = await client.get("/v1/results", headers=_INTERNAL_HEADERS)
    assert response.status_code == 200
    models = _models_in(response.json()["results"])
    assert ("deepgram", "nova-3") in models
    assert (_EA_PROVIDER, _EA_MODEL) in models


@pytest.mark.usefixtures("early_access_registry")
async def test_results_explicit_model_filter_stays_hidden(
    client: AsyncClient, postgresql: Any
) -> None:
    """Asking for the hidden model by name must not bypass the embargo."""
    await _seed_ea_and_public_rows(postgresql)

    response = await client.get(
        "/v1/results", params={"provider": _EA_PROVIDER, "model": _EA_MODEL}
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.usefixtures("early_access_registry")
async def test_leaderboard_embargo(client: AsyncClient, postgresql: Any) -> None:
    """/v1/leaderboard strips EARLY_ACCESS models for public callers only."""
    await _seed_ea_and_public_rows(postgresql)
    await _refresh_mv(postgresql)

    params = {"metric": "WER", "benchmark": "STT", "window": "24h"}

    public = await client.get("/v1/leaderboard", params=params)
    assert public.status_code == 200
    public_models = _models_in(public.json()["entries"])
    assert ("deepgram", "nova-3") in public_models
    assert (_EA_PROVIDER, _EA_MODEL) not in public_models

    internal = await client.get("/v1/leaderboard", params=params, headers=_INTERNAL_HEADERS)
    assert internal.status_code == 200
    assert (_EA_PROVIDER, _EA_MODEL) in _models_in(internal.json()["entries"])


@pytest.mark.usefixtures("early_access_registry")
async def test_aggregates_embargo_and_cache_isolation(client: AsyncClient, postgresql: Any) -> None:
    """/v1/results/aggregates: internal and public views never share a cache entry.

    The internal request goes first so a shared cache key would poison the
    public response with the hidden model.
    """
    await _seed_ea_and_public_rows(postgresql)
    await _refresh_mv(postgresql)
    await _fill_buckets(postgresql)

    params = {"benchmark": "STT", "window": "24h"}

    internal = await client.get("/v1/results/aggregates", params=params, headers=_INTERNAL_HEADERS)
    assert internal.status_code == 200
    internal_body = internal.json()
    assert (_EA_PROVIDER, _EA_MODEL) in _models_in(internal_body["model_stats"])
    assert (_EA_PROVIDER, _EA_MODEL) in _models_in(internal_body["series"])

    public = await client.get("/v1/results/aggregates", params=params)
    assert public.status_code == 200
    public_body = public.json()
    assert ("deepgram", "nova-3") in _models_in(public_body["model_stats"])
    assert (_EA_PROVIDER, _EA_MODEL) not in _models_in(public_body["model_stats"])
    assert (_EA_PROVIDER, _EA_MODEL) not in _models_in(public_body["series"])


@pytest.mark.usefixtures("early_access_registry")
async def test_providers_omits_early_access_from_public(client: AsyncClient) -> None:
    """Public /v1/providers must not even reveal an EARLY_ACCESS model's existence."""
    response = await client.get("/v1/providers")
    assert response.status_code == 200
    stt = response.json()["stt"]
    assert _EA_PROVIDER not in {p["provider"] for p in stt}


@pytest.mark.usefixtures("early_access_registry")
async def test_providers_serves_early_access_enabled_to_internal(client: AsyncClient) -> None:
    """Internal /v1/providers includes EARLY_ACCESS models, enabled."""
    response = await client.get("/v1/providers", headers=_INTERNAL_HEADERS)
    assert response.status_code == 200
    by_provider = {p["provider"]: p["models"] for p in response.json()["stt"]}
    assert _EA_PROVIDER in by_provider
    (model,) = by_provider[_EA_PROVIDER]
    assert model["model"] == _EA_MODEL
    assert model["disabled"] is False


def test_is_internal_requires_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No configured key means no request is internal, whatever it presents."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "stt-v1")
    monkeypatch.setenv("RUNNER_SHA", "test-sha")
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    settings = Settings()

    assert is_internal(x_internal_key="anything", settings=settings) is False
    assert is_internal(x_internal_key=None, settings=settings) is False

    monkeypatch.setenv("INTERNAL_API_KEY", "k")
    configured = Settings()
    assert is_internal(x_internal_key="k", settings=configured) is True
    assert is_internal(x_internal_key="wrong", settings=configured) is False


# ---------------------------------------------------------------------------
# Stealth models (env-defined aliases; see coval_bench.registries.stealth)
# ---------------------------------------------------------------------------

_STEALTH_ALIAS = "stealth-01"


@pytest.fixture
def stealth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Define one env-only stealth STT model before the app builds Settings."""
    monkeypatch.setenv(
        "STEALTH_MODELS",
        '{"stealth-01": {"benchmark": "STT", "provider": "acme", "model": "real-secret"}}',
    )


async def _seed_stealth_and_public_rows(postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, provider="stealth", model=_STEALTH_ALIAS)
    await _insert_result(postgresql, run_id, provider="deepgram", model="nova-3")


@pytest.mark.usefixtures("stealth_env")
async def test_results_hides_stealth_alias_from_public(
    client: AsyncClient, postgresql: Any
) -> None:
    """Alias rows are embargoed like registry EARLY_ACCESS models."""
    await _seed_stealth_and_public_rows(postgresql)

    for headers in ({}, _WRONG_HEADERS):
        response = await client.get("/v1/results", headers=headers)
        assert response.status_code == 200
        models = _models_in(response.json()["results"])
        assert ("deepgram", "nova-3") in models
        assert ("stealth", _STEALTH_ALIAS) not in models


@pytest.mark.usefixtures("stealth_env")
async def test_results_serves_stealth_alias_to_internal(
    client: AsyncClient, postgresql: Any
) -> None:
    await _seed_stealth_and_public_rows(postgresql)

    response = await client.get("/v1/results", headers=_INTERNAL_HEADERS)
    assert response.status_code == 200
    assert ("stealth", _STEALTH_ALIAS) in _models_in(response.json()["results"])


@pytest.mark.usefixtures("stealth_env")
async def test_providers_stealth_alias_internal_only(client: AsyncClient) -> None:
    """/v1/providers lists the alias (enabled) for internal callers only."""
    public = await client.get("/v1/providers")
    assert public.status_code == 200
    assert "stealth" not in {p["provider"] for p in public.json()["stt"]}

    internal = await client.get("/v1/providers", headers=_INTERNAL_HEADERS)
    assert internal.status_code == 200
    by_provider = {p["provider"]: p["models"] for p in internal.json()["stt"]}
    assert "stealth" in by_provider
    (model,) = by_provider["stealth"]
    assert model["model"] == _STEALTH_ALIAS
    assert model["disabled"] is False

    # Even the internal view carries only the alias — the real identity must
    # not appear anywhere in the serialized response.
    assert "real-secret" not in internal.text
    assert "acme" not in internal.text


async def test_stealth_rows_hidden_even_without_secret(
    client: AsyncClient, postgresql: Any
) -> None:
    """The stealth namespace is embargoed statically — a stale or unset
    STEALTH_MODELS on the API must hide alias rows, not leak them."""
    await _seed_stealth_and_public_rows(postgresql)
    await _refresh_mv(postgresql)

    public_results = await client.get("/v1/results")
    assert public_results.status_code == 200
    models = _models_in(public_results.json()["results"])
    assert ("deepgram", "nova-3") in models
    assert ("stealth", _STEALTH_ALIAS) not in models

    params = {"metric": "WER", "benchmark": "STT", "window": "24h"}
    public_board = await client.get("/v1/leaderboard", params=params)
    assert public_board.status_code == 200
    board_models = _models_in(public_board.json()["entries"])
    assert ("deepgram", "nova-3") in board_models
    assert ("stealth", _STEALTH_ALIAS) not in board_models

    internal = await client.get("/v1/results", headers=_INTERNAL_HEADERS)
    assert ("stealth", _STEALTH_ALIAS) in _models_in(internal.json()["results"])
