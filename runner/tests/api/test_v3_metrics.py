# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""/v3/metrics over HTTP."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from coval_bench.api.routers import v3_metrics
from coval_bench.config import Settings
from tests.api.conftest import COVAL_ORG, bearer

URL = "/v3/metrics"
ACME_ORG = "org_acme"


def _staff() -> dict[str, str]:
    return bearer(sub="user_staff", org_id=COVAL_ORG)


def _acme() -> dict[str, str]:
    return bearer(sub="user_acme", org_id=ACME_ORG)


async def _listed(client: AsyncClient, headers: dict[str, str] | None) -> dict[str, dict[str, Any]]:
    response = await client.get(URL, headers=headers)
    assert response.status_code == 200
    return {metric["code"]: metric for metric in response.json()["metrics"]}


async def test_each_caller_lists_the_metrics_it_may_see(client: AsyncClient) -> None:
    await client.post("/v3/companies", json={"slug": "acme", "name": "Acme"}, headers=_staff())
    await client.put("/v3/companies/acme/org", json={"org_id": ACME_ORG}, headers=_staff())
    patched = await client.patch(
        f"{URL}/CallLength", json={"visibility": "partner"}, headers=_staff()
    )
    assert patched.status_code == 200

    public = await _listed(client, None)
    assert {metric["visibility"] for metric in public.values()} == {"public"}
    assert public["WER"] == {
        "code": "WER",
        "display_name": "Word Error Rate",
        "visibility": "public",
        "collected": None,
    }

    partner = await _listed(client, _acme())
    assert partner.keys() == public.keys() | {"CallLength"}
    assert partner["CallLength"]["collected"] is None

    staff = await _listed(client, _staff())
    hidden = staff.keys() - partner.keys()
    assert hidden
    assert {staff[code]["visibility"] for code in hidden} == {"coval"}
    assert staff["WER"]["collected"] is True
    assert staff["ExpectedBehaviorAdherence"]["collected"] is False


async def test_only_staff_change_a_metric(client: AsyncClient) -> None:
    for headers in (None, _acme()):
        refused = await client.patch(f"{URL}/WER", json={"collected": False}, headers=headers)
        assert (refused.status_code, refused.json()["error"]["code"]) == (403, "forbidden")

    missing = await client.patch(f"{URL}/Nope", json={"collected": False}, headers=_staff())
    assert missing.status_code == 404
    for body in ({}, {"visibility": "secret"}, {"code": "Renamed"}):
        assert (await client.patch(f"{URL}/WER", json=body, headers=_staff())).status_code == 400

    off = await client.patch(f"{URL}/WER", json={"collected": False}, headers=_staff())
    assert off.status_code == 200
    assert off.headers["Cache-Control"] == "private, no-store"
    assert off.json() == {
        "code": "WER",
        "display_name": "Word Error Rate",
        "visibility": "public",
        "collected": False,
    }

    both = await client.patch(
        f"{URL}/WER", json={"visibility": "coval", "collected": True}, headers=_staff()
    )
    assert (both.json()["visibility"], both.json()["collected"]) == ("coval", True)
    assert "WER" not in await _listed(client, None)


async def test_a_collected_change_starts_the_sync(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Settings] = []

    async def start(settings: Settings) -> bool:
        started.append(settings)
        return True

    monkeypatch.setattr(v3_metrics, "_start_sync", start)

    await client.patch(f"{URL}/V2V", json={"visibility": "partner"}, headers=_staff())
    await client.patch(f"{URL}/V2V", json={"collected": True}, headers=_staff())
    await client.patch(f"{URL}/TTFA", json={"collected": False}, headers=_staff())
    assert started == []

    await client.patch(f"{URL}/V2V", json={"collected": False}, headers=_staff())
    assert len(started) == 1
