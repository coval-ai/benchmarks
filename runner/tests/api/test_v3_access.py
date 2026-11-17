# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who sees an unpublished model on /v3: decided by the companies table alone."""

from __future__ import annotations

from typing import Any

import psycopg
from httpx import AsyncClient

from coval_bench.registries import RegisteredModel
from coval_bench.registries.benchmarks import Benchmark
from tests.api.conftest import (
    COVAL_ORG,
    EA_MODEL,
    EA_ORG,
    EA_PROVIDER,
    _make_db_url,
    add_models,
    bearer,
    set_company_org,
)

LINKED = "org_test_linked"


def _unpublished(provider: str, model: str) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.STT, provider=provider, model=model, collected=True, published=False
    )


async def _providers(client: AsyncClient, org_id: str | None) -> set[str]:
    headers = {} if org_id is None else bearer(sub="u", org_id=org_id)
    response = await client.get("/v3/latency", headers=headers)
    assert response.status_code == 200
    return {row["provider"] for row in response.json()["latency"]}


async def test_an_org_sees_its_own_companys_unpublished_models_and_no_one_elses(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(postgresql, _unpublished(EA_PROVIDER, EA_MODEL), _unpublished("rival", "next"))
    set_company_org(postgresql, EA_PROVIDER, LINKED)
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        for (model_id,) in conn.execute("SELECT id FROM benchmarks_v2.models").fetchall():
            conn.execute(
                "INSERT INTO benchmarks_v2.probe_results"
                " (model_id, scheduled_at, sent_at, status, ttft_ms, ttfa_ms)"
                " VALUES (%s, now() - interval '1 hour', now() - interval '1 hour',"
                " 'succeeded', 100, 100)",
                (model_id,),
            )

    public = await _providers(client, None)
    linked = await _providers(client, LINKED)
    settings_only = await _providers(client, EA_ORG)
    staff = await _providers(client, COVAL_ORG)

    assert {EA_PROVIDER, "rival"}.isdisjoint(public)
    assert linked == public | {EA_PROVIDER}
    assert settings_only == public
    assert staff == public | {EA_PROVIDER, "rival"}
