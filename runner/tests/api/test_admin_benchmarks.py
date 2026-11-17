# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The admin benchmark registry endpoints over HTTP."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from tests.api.conftest import COVAL_ORG, EA_ORG, bearer

ADMIN = bearer(sub="user_admin", org_id=COVAL_ORG, email="admin@coval.dev")
URL = "/v1/admin/benchmarks"


async def _create(client: AsyncClient, slug: str = "stt-wildasr", **overrides: Any) -> Any:
    body = {"slug": slug, "modality": "STT", "name": "WildASR", **overrides}
    return await client.post(URL, json=body, headers=ADMIN)


async def _put(client: AsyncClient, slug: str, dataset_id: str, condition: str | None) -> Any:
    return await client.put(
        f"{URL}/{slug}/datasets/{dataset_id}", json={"condition": condition}, headers=ADMIN
    )


async def test_admin_routes_require_a_coval_token(client: AsyncClient) -> None:
    assert (await client.get(URL)).status_code == 401
    assert (await client.get(URL, headers=bearer(sub="u", org_id=EA_ORG))).status_code == 403
    assert (await client.post(URL, json={})).status_code == 401


async def test_a_benchmark_is_created_and_listed(client: AsyncClient) -> None:
    created = await _create(client)
    assert created.status_code == 201
    assert created.headers["Cache-Control"] == "private, no-store"
    assert created.json()["updated_by_email"] == "admin@coval.dev"

    assert (await _create(client)).status_code == 409
    assert (await _create(client, slug="Bad Slug")).status_code == 422

    listed = (await client.get(URL, headers=ADMIN)).json()["benchmarks"]
    assert [(b["slug"], b["modality"], b["name"], b["datasets"]) for b in listed] == [
        ("stt-wildasr", "STT", "WildASR", [])
    ]


async def test_datasets_are_added_relabelled_and_kept_to_one_per_condition(
    client: AsyncClient,
) -> None:
    await _create(client)
    await _create(client, slug="stt-v3", name="STT v3")

    assert (await _put(client, "nope", "stt-wildasr-clean", "clean")).status_code == 404
    assert (await _put(client, "stt-wildasr", "stt-wildasr-clean", "clean")).status_code == 200
    added = await _put(client, "stt-wildasr", "stt-wildasr-farfield", "far")
    assert added.status_code == 200

    relabelled = await _put(client, "stt-wildasr", "stt-wildasr-farfield", "farfield")
    assert relabelled.json()["datasets"] == [
        {"dataset_id": "stt-wildasr-clean", "condition": "clean"},
        {"dataset_id": "stt-wildasr-farfield", "condition": "farfield"},
    ]

    taken = await _put(client, "stt-wildasr", "stt-wildasr-other", "clean")
    assert taken.status_code == 409
    elsewhere = await _put(client, "stt-v3", "stt-wildasr-clean", None)
    assert elsewhere.status_code == 409

    assert (await _put(client, "stt-v3", "stt-v3", None)).status_code == 200
    assert (await _put(client, "stt-v3", "stt-v3-copy", None)).status_code == 409


async def test_a_dataset_is_removed_from_the_benchmark_that_owns_it(client: AsyncClient) -> None:
    await _create(client)
    await _create(client, slug="stt-v3", name="STT v3")
    await _put(client, "stt-wildasr", "stt-wildasr-clean", "clean")

    wrong = await client.delete(f"{URL}/stt-v3/datasets/stt-wildasr-clean", headers=ADMIN)
    assert wrong.status_code == 404

    removed = await client.delete(f"{URL}/stt-wildasr/datasets/stt-wildasr-clean", headers=ADMIN)
    assert removed.json()["datasets"] == []
