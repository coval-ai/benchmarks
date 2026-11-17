# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The admin company endpoints over HTTP."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from tests.api.conftest import COVAL_ORG, EA_ORG, bearer

ADMIN = bearer(sub="user_admin", org_id=COVAL_ORG, email="admin@coval.dev")
URL = "/v1/admin/companies"


async def _create(client: AsyncClient, slug: str = "acme") -> Any:
    return await client.post(URL, json={"slug": slug, "name": slug.title()}, headers=ADMIN)


async def _set_org(client: AsyncClient, slug: str, org_id: str) -> Any:
    return await client.put(f"{URL}/{slug}/org", json={"org_id": org_id}, headers=ADMIN)


async def test_admin_routes_require_a_coval_token(client: AsyncClient) -> None:
    assert (await client.get(URL)).status_code == 401
    assert (await client.get(URL, headers=bearer(sub="u", org_id=EA_ORG))).status_code == 403
    assert (await client.put(f"{URL}/acme/org", json={"org_id": "org_1"})).status_code == 401


async def test_a_company_is_created_and_listed_privately(client: AsyncClient) -> None:
    created = await _create(client)
    assert created.status_code == 201
    assert created.headers["Cache-Control"] == "private, no-store"
    assert (created.json()["org_id"], created.json()["updated_by_email"]) == (
        None,
        "admin@coval.dev",
    )
    assert (await _create(client)).status_code == 409

    listed = (await client.get(URL, headers=ADMIN)).json()["companies"]
    assert [(c["slug"], c["name"]) for c in listed] == [("acme", "Acme")]


async def test_an_org_is_set_on_one_company_and_cleared(client: AsyncClient) -> None:
    await _create(client)
    await _create(client, "other")

    assert (await _set_org(client, "nope", "org_1")).status_code == 404
    assert (await _set_org(client, "acme", "not-an-org")).status_code == 422

    assert (await _set_org(client, "acme", "org_1")).json()["org_id"] == "org_1"
    assert (await _set_org(client, "other", "org_1")).status_code == 409
    assert (await _set_org(client, "acme", "org_2")).json()["org_id"] == "org_2"

    cleared = await client.delete(f"{URL}/acme/org", headers=ADMIN)
    assert cleared.json()["org_id"] is None
    assert (await _set_org(client, "other", "org_2")).status_code == 200
