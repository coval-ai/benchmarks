# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""/v3/companies over HTTP."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from httpx import AsyncClient

from coval_bench.registries import RegisteredModel, Visibility
from coval_bench.registries.benchmarks import Benchmark
from tests.api.conftest import COVAL_ORG, _make_db_url, add_models, bearer

URL = "/v3/companies"
ACME_ORG = "org_acme"


def _staff() -> dict[str, str]:
    return bearer(sub="user_staff", org_id=COVAL_ORG)


def _acme() -> dict[str, str]:
    return bearer(sub="user_acme", org_id=ACME_ORG)


@pytest.fixture(autouse=True)
def _empty_registry(postgresql: Any) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute("DELETE FROM benchmarks_v2.model_history")
        conn.execute("DELETE FROM benchmarks_v2.models")
        conn.execute("DELETE FROM benchmarks_v2.companies")


def _model(provider: str, model: str, visibility: Visibility) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.STT,
        provider=provider,
        model=model,
        collected=True,
        visibility=visibility,
    )


async def _create(client: AsyncClient, slug: str) -> Any:
    return await client.post(URL, json={"slug": slug, "name": slug.title()}, headers=_staff())


async def _listed(client: AsyncClient, headers: dict[str, str] | None) -> list[dict[str, Any]]:
    response = await client.get(URL, headers=headers)
    assert response.status_code == 200
    return list(response.json()["companies"])


async def test_a_company_is_listed_once_one_of_its_models_is_seen(
    client: AsyncClient, postgresql: Any
) -> None:
    for slug in ("acme", "quiet", "rival"):
        await _create(client, slug)
    await client.put(f"{URL}/quiet/org", json={"org_id": ACME_ORG}, headers=_staff())
    add_models(
        postgresql,
        _model("acme", "open", "public"),
        _model("quiet", "early", "partner"),
        _model("rival", "early", "partner"),
    )

    assert await _listed(client, None) == [
        {"slug": "acme", "name": "Acme", "org_id": None, "updated_by_org": None, "updated_at": None}
    ]
    assert [company["slug"] for company in await _listed(client, _acme())] == ["acme", "quiet"]

    staff = await _listed(client, _staff())
    assert [(c["slug"], c["org_id"], c["updated_by_org"]) for c in staff] == [
        ("acme", None, COVAL_ORG),
        ("quiet", ACME_ORG, COVAL_ORG),
        ("rival", None, COVAL_ORG),
    ]


async def test_only_staff_write_companies(client: AsyncClient) -> None:
    for headers in (None, _acme()):
        refused = await client.post(URL, json={"slug": "acme", "name": "Acme"}, headers=headers)
        assert (refused.status_code, refused.json()["error"]["code"]) == (403, "forbidden")

    created = await _create(client, "acme")
    assert created.status_code == 201
    assert created.headers["Cache-Control"] == "private, no-store"
    conflict = await _create(client, "acme")
    assert (conflict.status_code, conflict.json()["error"]["code"]) == (409, "conflict")

    org = {"org_id": "org_1"}
    assert (await client.put(f"{URL}/acme/org", json=org, headers=_acme())).status_code == 403
    assert (await client.put(f"{URL}/nope/org", json=org, headers=_staff())).status_code == 404
    assert (await client.put(f"{URL}/acme/org", json=org, headers=_staff())).json()[
        "org_id"
    ] == "org_1"
    assert (await client.delete(f"{URL}/acme/org", headers=_acme())).status_code == 403
    cleared = await client.delete(f"{URL}/acme/org", headers=_staff())
    assert cleared.json()["org_id"] is None
