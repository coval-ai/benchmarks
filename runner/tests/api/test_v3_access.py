# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who sees an unpublished model on /v3: decided by the companies table alone."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from httpx import AsyncClient

from coval_bench.api.v3_access import Caller, can_see
from coval_bench.registries import RegisteredModel, Visibility
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
        benchmark=Benchmark.STT,
        provider=provider,
        model=model,
        collected=True,
        visibility="partner",
    )


def _probe_every_model(postgresql: Any) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        for (model_id,) in conn.execute("SELECT id FROM benchmarks_v2.models").fetchall():
            conn.execute(
                "INSERT INTO benchmarks_v2.probe_results"
                " (model_id, scheduled_at, sent_at, status, ttft_ms, ttfa_ms)"
                " VALUES (%s, now() - interval '1 hour', now() - interval '1 hour',"
                " 'succeeded', 100, 100)",
                (model_id,),
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
    _probe_every_model(postgresql)

    public = await _providers(client, None)
    linked = await _providers(client, LINKED)
    settings_only = await _providers(client, EA_ORG)
    staff = await _providers(client, COVAL_ORG)

    assert {EA_PROVIDER, "rival"}.isdisjoint(public)
    assert linked == public | {EA_PROVIDER}
    assert settings_only == public
    assert staff == public | {EA_PROVIDER, "rival"}


async def test_a_coval_level_model_is_hidden_from_its_own_company(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(
        postgresql,
        RegisteredModel(
            benchmark=Benchmark.STT,
            provider=EA_PROVIDER,
            model=EA_MODEL,
            collected=True,
            visibility="coval",
        ),
    )
    set_company_org(postgresql, EA_PROVIDER, LINKED)
    _probe_every_model(postgresql)

    assert EA_PROVIDER not in await _providers(client, LINKED)
    assert EA_PROVIDER in await _providers(client, COVAL_ORG)


async def test_a_models_level_is_not_taken_from_its_namesake_in_another_modality(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(
        postgresql,
        RegisteredModel(
            benchmark=Benchmark.STT,
            provider=EA_PROVIDER,
            model="omni",
            collected=True,
            visibility="coval",
        ),
        RegisteredModel(
            benchmark=Benchmark.TTS,
            provider=EA_PROVIDER,
            model="omni",
            voice="v",
            collected=True,
            visibility="public",
        ),
    )
    _probe_every_model(postgresql)

    stt = (await client.get("/v3/latency?modality=STT")).json()["latency"]
    tts = (await client.get("/v3/latency?modality=TTS")).json()["latency"]

    assert EA_PROVIDER not in {row["provider"] for row in stt}
    assert EA_PROVIDER in {row["provider"] for row in tts}


async def test_a_partner_level_metric_shows_only_on_the_callers_own_models(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(
        postgresql,
        RegisteredModel(
            benchmark=Benchmark.STT,
            provider=EA_PROVIDER,
            model=EA_MODEL,
            collected=True,
            visibility="public",
        ),
    )
    set_company_org(postgresql, EA_PROVIDER, LINKED)
    _probe_every_model(postgresql)
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute("UPDATE benchmarks_v2.metrics SET visibility = 'partner'")

    assert await _providers(client, None) == set()
    assert await _providers(client, LINKED) == {EA_PROVIDER}
    assert await _providers(client, COVAL_ORG) > {EA_PROVIDER}


ANONYMOUS = Caller("absent")
OWNER = Caller("accepted", "partner", "acme")
RIVAL = Caller("accepted", "partner", "rival")
STAFF = Caller("accepted", "coval")


@pytest.mark.parametrize(
    ("level", "who_sees_it"),
    [
        ("public", {ANONYMOUS, OWNER, RIVAL, STAFF}),
        ("partner", {OWNER, STAFF}),
        ("coval", {STAFF}),
    ],
)
def test_who_sees_a_result_on_acmes_model(level: Visibility, who_sees_it: set[Caller]) -> None:
    callers = (ANONYMOUS, OWNER, RIVAL, STAFF)
    assert {who for who in callers if can_see(who, level, "acme")} == who_sees_it
