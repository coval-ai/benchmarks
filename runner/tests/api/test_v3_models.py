# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""/v3/models over HTTP."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from httpx import AsyncClient

from coval_bench.api.routers import admin_models
from tests.api.conftest import COVAL_ORG, _make_db_url, bearer, set_company_org

ACME_ORG = "org_acme"


def _staff() -> dict[str, str]:
    return bearer(sub="user_staff", org_id=COVAL_ORG)


def _acme() -> dict[str, str]:
    return bearer(sub="user_acme", org_id=ACME_ORG)


def _stranger() -> dict[str, str]:
    return bearer(sub="user_other", org_id="org_other")


@pytest.fixture(autouse=True)
def _empty_registry(postgresql: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute("DELETE FROM benchmarks_v2.model_history")
        conn.execute("DELETE FROM benchmarks_v2.models")
        conn.execute("DELETE FROM benchmarks_v2.companies")
    monkeypatch.setattr(admin_models, "_implemented_providers", lambda modality: None)


async def _model(client: AsyncClient, provider: str, model: str, visibility: str) -> Any:
    body = {"modality": "STT", "provider": provider, "model": model, "visibility": visibility}
    return await client.post("/v3/models", json=body, headers=_staff())


async def _roster(client: AsyncClient, postgresql: Any) -> None:
    """Acme, whose org is ACME_ORG, with a public and a partner model, and a hidden rival."""
    set_company_org(postgresql, "acme", ACME_ORG)
    await _model(client, "acme", "open", "public")
    await _model(client, "acme", "early", "partner")
    await _model(client, "rival", "early", "partner")


async def _models(client: AsyncClient, headers: dict[str, str] | None) -> list[dict[str, Any]]:
    response = await client.get("/v3/models", headers=headers)
    assert response.status_code == 200
    return list(response.json()["models"])


def _names(models: list[dict[str, Any]]) -> list[str]:
    return [f"{model['provider']}/{model['model']}" for model in models]


async def test_each_caller_lists_the_models_it_may_see(
    client: AsyncClient, postgresql: Any
) -> None:
    await _roster(client, postgresql)

    public = await _models(client, None)
    assert _names(public) == ["acme/open"]
    assert (public[0]["collected"], public[0]["updated_by_org"], public[0]["updated_at"]) == (
        None,
        None,
        None,
    )
    assert _names(await _models(client, _stranger())) == ["acme/open"]
    assert _names(await _models(client, _acme())) == ["acme/open", "acme/early"]

    staff = await _models(client, _staff())
    assert _names(staff) == ["acme/open", "acme/early", "rival/early"]
    assert (staff[0]["collected"], staff[0]["updated_by_org"]) == (True, COVAL_ORG)


async def test_only_staff_write_models_and_the_org_is_recorded(
    client: AsyncClient, postgresql: Any
) -> None:
    body = {"modality": "STT", "provider": "acme", "model": "open"}
    for headers in (None, _acme()):
        refused = await client.post("/v3/models", json=body, headers=headers)
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "forbidden"

    unruly = await client.post(
        "/v3/models", json={**body, "architecture": "cascaded"}, headers=_staff()
    )
    assert (unruly.status_code, unruly.json()["error"]["code"]) == (400, "invalid_request")

    created = await _model(client, "acme", "open", "partner")
    assert created.status_code == 201
    assert created.headers["Cache-Control"] == "private, no-store"
    model_id = created.json()["id"]
    url = f"/v3/models/{model_id}"

    edit = {"display_name": "Acme Open"}
    assert (await client.patch(url, json=edit, headers=_acme())).status_code == 403
    assert (await client.patch(url, json=edit, headers=_staff())).status_code == 428
    patched = await client.patch(
        url, json=edit, headers={**_staff(), "If-Match": created.headers["ETag"]}
    )
    assert patched.status_code == 200
    assert patched.json()["model"]["display_name"] == "Acme Open"

    stale = await client.patch(
        url, json=edit, headers={**_staff(), "If-Match": created.headers["ETag"]}
    )
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "precondition_failed"

    with psycopg.connect(_make_db_url(postgresql)) as conn:
        orgs = conn.execute(
            "SELECT DISTINCT changed_by_org_id FROM benchmarks_v2.model_history"
            " WHERE model_id = %s",
            (model_id,),
        ).fetchall()
    assert orgs == [(COVAL_ORG,)]
