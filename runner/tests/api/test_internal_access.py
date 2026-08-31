# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the early-access embargo over HTTP.

EARLY_ACCESS models must never appear in public responses of the data
endpoints. The one proof is a Clerk bearer session token: the coval org sees
them everywhere, and a mapped provider org sees what its grant names.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from coval_bench.api.internal import VARY_HEADERS
from coval_bench.registries import Benchmark, RegisteredModel
from tests.api.conftest import (
    COVAL_ORG,
    EA_MODEL,
    EA_MODEL_OTHER,
    EA_ORG,
    EA_ORG_OTHER,
    EA_PROVIDER,
    _fill_buckets,
    _insert_result,
    _insert_run,
    _refresh_mv,
    add_models,
    bearer,
)

# A token the stubbed Clerk instance never minted: an unknown proof, not an absent one.
_WRONG_HEADERS = {"Authorization": "Bearer not-a-real-token"}

_EA_PROVIDER = EA_PROVIDER
_EA_MODEL = EA_MODEL


def _internal_headers() -> dict[str, str]:
    return bearer(org_id=COVAL_ORG)


@pytest.fixture
def early_access_registry(postgresql: Any) -> None:
    """Add two EARLY_ACCESS STT models to this test's registry tables.

    Both sit on one provider so a per-model grant can be told apart from a
    per-provider one.
    """
    add_models(
        postgresql,
        *(
            RegisteredModel(
                benchmark=Benchmark.STT,
                provider=_EA_PROVIDER,
                model=model,
                collected=True,
                published=False,
            )
            for model in (_EA_MODEL, EA_MODEL_OTHER)
        ),
    )


async def _seed_ea_and_public_rows(postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, provider=_EA_PROVIDER, model=_EA_MODEL)
    await _insert_result(postgresql, run_id, provider="deepgram", model="nova-3")


def _models_in(results: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(r["provider"], r["model"]) for r in results}


@pytest.mark.usefixtures("early_access_registry")
async def test_results_hides_early_access_from_public(client: AsyncClient, postgresql: Any) -> None:
    """Public and bad-bearer callers never see EARLY_ACCESS rows on /v1/results."""
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
    """A coval org session unlocks EARLY_ACCESS rows on /v1/results."""
    await _seed_ea_and_public_rows(postgresql)

    response = await client.get("/v1/results", headers=_internal_headers())
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

    internal = await client.get("/v1/leaderboard", params=params, headers=_internal_headers())
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

    internal = await client.get(
        "/v1/results/aggregates", params=params, headers=_internal_headers()
    )
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
async def test_timeline_embargo_and_cache_isolation(client: AsyncClient, postgresql: Any) -> None:
    """Timeline cache entries keep the caller's embargo scope, like aggregates."""
    await _seed_ea_and_public_rows(postgresql)
    await _fill_buckets(postgresql)
    params = {"benchmark": "STT", "window": "24h"}

    internal = await client.get("/v1/results/timeline", params=params, headers=_internal_headers())
    assert internal.status_code == 200
    assert (_EA_PROVIDER, _EA_MODEL) in _models_in(internal.json()["points"])

    public = await client.get("/v1/results/timeline", params=params)
    assert public.status_code == 200
    assert (_EA_PROVIDER, _EA_MODEL) not in _models_in(public.json()["points"])


@pytest.mark.usefixtures("early_access_registry")
async def test_aggregates_by_dataset_embargo_and_cache_isolation(
    client: AsyncClient, postgresql: Any
) -> None:
    """/v1/results/aggregates/by-dataset: internal and public views never share
    a cache entry.

    The internal request goes first so a shared cache key would poison the
    public response with the hidden model.
    """
    await _seed_ea_and_public_rows(postgresql)
    await _refresh_mv(postgresql)

    params = {"benchmark": "STT", "window": "24h"}

    def stats_models(body: dict[str, Any]) -> set[tuple[str, str]]:
        return _models_in([s for block in body["blocks"] for s in block["model_stats"]])

    internal = await client.get(
        "/v1/results/aggregates/by-dataset", params=params, headers=_internal_headers()
    )
    assert internal.status_code == 200
    assert (_EA_PROVIDER, _EA_MODEL) in stats_models(internal.json())

    public = await client.get("/v1/results/aggregates/by-dataset", params=params)
    assert public.status_code == 200
    public_models = stats_models(public.json())
    assert ("deepgram", "nova-3") in public_models
    assert (_EA_PROVIDER, _EA_MODEL) not in public_models


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
    response = await client.get("/v1/providers", headers=_internal_headers())
    assert response.status_code == 200
    by_provider = {p["provider"]: p["models"] for p in response.json()["stt"]}
    assert _EA_PROVIDER in by_provider
    models = {m["model"]: m for m in by_provider[_EA_PROVIDER]}
    assert models.keys() == {_EA_MODEL, EA_MODEL_OTHER}
    assert all(m["disabled"] is False for m in models.values())


@pytest.mark.usefixtures("early_access_registry")
async def test_the_retired_x_headers_prove_nothing(client: AsyncClient) -> None:
    """The removed X-Internal-Key / X-EA-Token headers are dead: ignored, not honoured."""
    response = await client.get(
        "/v1/providers",
        headers={"X-Internal-Key": "test-internal-key", "X-EA-Token": "test-ea-token"},
    )
    assert response.status_code == 200
    assert response.headers["X-EA-Token-Status"] == "absent"
    assert _EA_PROVIDER not in {p["provider"] for p in response.json()["stt"]}


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/v1/providers", None),
        ("/v1/results", None),
        ("/v1/leaderboard", {"metric": "WER", "benchmark": "STT"}),
        ("/v1/results/aggregates", {"benchmark": "STT"}),
        ("/v1/results/timeline", {"benchmark": "STT"}),
        ("/v1/results/aggregates/by-dataset", {"benchmark": "STT"}),
    ],
)
async def test_vary_lists_the_proof_header(
    client: AsyncClient, path: str, params: dict[str, str] | None
) -> None:
    """Every embargo-gated endpoint varies on the bearer proof and nothing retired."""
    response = await client.get(path, params=params)
    assert response.status_code == 200
    vary = response.headers["Vary"]
    for header in VARY_HEADERS.split(", "):
        assert header in vary, f"{header} missing from Vary: {vary!r}"
    assert "x-internal-key" not in vary.lower()
    assert "x-ea-token" not in vary.lower()


async def test_vary_keeps_the_proof_header_once_gzip_appends(client: AsyncClient) -> None:
    """SelectiveGZipMiddleware appends Accept-Encoding after the route returns.

    /v1/providers serves the whole registry, so it clears the 1024-byte gzip
    threshold — an assignment in the dependency would be lost here.
    """
    response = await client.get("/v1/providers", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    vary = response.headers["Vary"]
    for header in (*VARY_HEADERS.split(", "), "Accept-Encoding"):
        assert header in vary, f"{header} missing from Vary: {vary!r}"


async def test_public_responses_are_cacheable_privileged_ones_are_not(
    client: AsyncClient,
) -> None:
    """The embargo gate must mark a privileged response no-store."""
    public = await client.get("/v1/providers")
    assert public.headers.get("Cache-Control") is None

    internal = await client.get("/v1/providers", headers=_internal_headers())
    assert internal.headers["Cache-Control"] == "private, no-store"


@pytest.mark.usefixtures("early_access_registry")
async def test_two_org_grants_never_share_an_aggregates_cache_entry(
    client: AsyncClient, postgresql: Any
) -> None:
    """Org A then org B on the same cache: B must not see A's model.

    This is the case that actually proves the cache key. Only /v1/results/aggregates
    caches, and A goes first so a key that ignored the caller would serve A's rows
    straight back to B.
    """
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, provider=_EA_PROVIDER, model=_EA_MODEL)
    await _insert_result(postgresql, run_id, provider=_EA_PROVIDER, model=EA_MODEL_OTHER)
    await _refresh_mv(postgresql)
    await _fill_buckets(postgresql)

    params = {"benchmark": "STT", "window": "24h"}

    first = await client.get("/v1/results/aggregates", params=params, headers=bearer(org_id=EA_ORG))
    assert first.status_code == 200
    assert first.headers["X-EA-Token-Status"] == "accepted"
    assert (_EA_PROVIDER, _EA_MODEL) in _models_in(first.json()["model_stats"])
    assert (_EA_PROVIDER, EA_MODEL_OTHER) not in _models_in(first.json()["model_stats"])

    second = await client.get(
        "/v1/results/aggregates", params=params, headers=bearer(org_id=EA_ORG_OTHER)
    )
    assert second.status_code == 200
    stats = _models_in(second.json()["model_stats"])
    assert (_EA_PROVIDER, EA_MODEL_OTHER) in stats
    assert (_EA_PROVIDER, _EA_MODEL) not in stats, "org A's model leaked into org B's view"


@pytest.mark.usefixtures("early_access_registry")
async def test_org_grant_scopes_the_providers_catalogue(client: AsyncClient) -> None:
    """A grant reveals its own model and keeps the sibling on the same provider hidden."""
    response = await client.get("/v1/providers", headers=bearer(org_id=EA_ORG))
    assert response.status_code == 200
    by_provider = {p["provider"]: p["models"] for p in response.json()["stt"]}
    assert {m["model"] for m in by_provider[_EA_PROVIDER]} == {_EA_MODEL}
