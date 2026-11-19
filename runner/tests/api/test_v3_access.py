# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who sees an unpublished model on /v3: decided by the companies table alone."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import psycopg
import pytest
from clerk_backend_api import models
from fastapi import Depends, FastAPI, HTTPException
from httpx import AsyncClient
from pydantic import SecretStr

from coval_bench.api import clerk
from coval_bench.api.v3_access import Caller, can_see, coval_only, viewer
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
_STAFF_ONLY = Depends(coval_only)
_ANY_READER = Depends(viewer)


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


async def test_an_orgs_api_key_gets_the_same_view_as_its_session(
    client: AsyncClient, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_models(postgresql, _unpublished(EA_PROVIDER, EA_MODEL))
    set_company_org(postgresql, EA_PROVIDER, LINKED)
    _probe_every_model(postgresql)

    async def owner(key: str, secret: Any) -> str:
        if key == "ak_down":
            raise ConnectionError
        if key != "ak_linked":
            raise clerk.UnknownApiKey
        return LINKED

    monkeypatch.setattr(clerk, "api_key_owner", owner)

    async def get(key: str) -> Any:
        return await client.get("/v3/latency", headers={"Authorization": f"Bearer {key}"})

    linked, stranger, down = await get("ak_linked"), await get("ak_other"), await get("ak_down")

    assert EA_PROVIDER in {row["provider"] for row in linked.json()["latency"]}
    assert linked.headers["Cache-Control"] == "private, no-store"
    assert EA_PROVIDER not in {row["provider"] for row in stranger.json()["latency"]}
    assert down.status_code == 503


async def test_a_good_api_key_is_remembered_and_a_rejected_one_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    async def verify(*, secret: str) -> Any:
        asked.append(secret)
        if secret not in {"ak_good"}:
            raise models.VerifyAPIKeyAPIKeysResponseResponseBody(
                models.VerifyAPIKeyAPIKeysResponseResponseBodyData(errors=[]),
                httpx.Response(404, request=httpx.Request("POST", "https://clerk.test")),
            )
        return SimpleNamespace(subject="org_1", revoked=False, expired=False)

    backend = SimpleNamespace(api_keys=SimpleNamespace(verify_api_key_async=verify))
    monkeypatch.setattr(clerk, "_backend", lambda secret: backend)
    monkeypatch.setattr(clerk, "_API_KEY_OWNERS", {})
    secret = SecretStr("sk_test_not_real")

    assert await clerk.api_key_owner("ak_good", secret) == "org_1"
    assert await clerk.api_key_owner("ak_good", secret) == "org_1"
    with pytest.raises(clerk.UnknownApiKey):
        await clerk.api_key_owner("ak_bad", secret)
    assert asked == ["ak_good", "ak_bad"]


async def test_a_route_learns_the_callers_org_and_refuses_in_the_v3_error_shape(
    app: FastAPI, client: AsyncClient
) -> None:
    @app.get("/v3/test-staff")
    async def staff(who: Caller = _STAFF_ONLY) -> dict[str, Any]:
        return {"org": who.org_id}

    @app.get("/v3/test-read")
    async def read(who: Caller = _ANY_READER) -> dict[str, Any]:
        return {"org": who.org_id}

    @app.get("/v3/test-refused/{status}")
    async def refused(status: int) -> None:
        raise HTTPException(status, "no")

    coval = bearer(sub="u", org_id=COVAL_ORG)
    assert (await client.get("/v3/test-staff", headers=coval)).json() == {"org": COVAL_ORG}

    anonymous = await client.get("/v3/test-read")
    assert anonymous.json() == {"org": None}
    assert anonymous.headers["Cache-Control"].startswith("public")
    signed_in = await client.get("/v3/test-read", headers=bearer(sub="u", org_id=EA_ORG))
    assert signed_in.json() == {"org": EA_ORG}
    assert signed_in.headers["Cache-Control"] == "private, no-store"

    async def answer(status: int) -> tuple[int, str]:
        response = await client.get(f"/v3/test-refused/{status}")
        return response.status_code, response.json()["error"]["code"]

    assert await answer(422) == (400, "invalid_request")
    assert await answer(409) == (409, "conflict")
    assert await answer(412) == (412, "precondition_failed")
    assert await answer(428) == (428, "precondition_required")
