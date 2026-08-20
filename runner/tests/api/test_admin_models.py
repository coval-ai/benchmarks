# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The admin model-state endpoints: coval-only reads and toggles.

The write path is the one place an outside party could flip what runs and what
the public sees, so the auth tests enumerate every non-coval caller class.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from coval_bench.registries import MODEL_REGISTRY
from tests.api.conftest import EA_ORG, INTERNAL_EMAIL, bearer

_NOVA3 = "/v1/admin/models/STT/deepgram/nova-3"
_PATCH = {"running": False, "shown": True}


def _internal() -> dict[str, str]:
    return bearer(email=INTERNAL_EMAIL)


@pytest.mark.parametrize(
    "headers",
    [
        {},  # anonymous
        {"Authorization": "Bearer not-a-real-token"},  # unverifiable
        {"Authorization": "Basic abc"},  # wrong scheme
    ],
    ids=["anonymous", "bad-token", "wrong-scheme"],
)
async def test_admin_endpoints_403_without_internal_proof(
    client: AsyncClient, headers: dict[str, str]
) -> None:
    assert (await client.get("/v1/admin/models", headers=headers)).status_code == 403
    assert (await client.patch(_NOVA3, json=_PATCH, headers=headers)).status_code == 403


async def test_partner_token_is_403(client: AsyncClient) -> None:
    """A valid partner token grants data visibility, never admin."""
    headers = bearer(org_id=EA_ORG, email="partner@example.com")
    assert (await client.get("/v1/admin/models", headers=headers)).status_code == 403
    assert (await client.patch(_NOVA3, json=_PATCH, headers=headers)).status_code == 403


async def test_internal_email_beats_active_partner_org(client: AsyncClient) -> None:
    """A coval dev previewing a partner org's data view can still administer."""
    headers = bearer(org_id=EA_ORG, email=INTERNAL_EMAIL)
    assert (await client.get("/v1/admin/models", headers=headers)).status_code == 200


async def test_get_lists_every_registry_model_with_state(client: AsyncClient) -> None:
    response = await client.get("/v1/admin/models", headers=_internal())
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) == len(MODEL_REGISTRY)
    by_key = {(m["benchmark"], m["provider"], m["model"]): m for m in models}
    nova = by_key[("STT", "deepgram", "nova-3")]
    assert nova["state"]["running"] is True
    assert nova["state"]["shown"] is True
    # Seeded from the migration snapshot: an embargoed model reads Hidden.
    qwen = by_key[("TTS", "baseten", "qwen3-tts-1.7b")]
    assert qwen["state"]["running"] is True
    assert qwen["state"]["shown"] is False


async def test_patch_updates_state_and_history(client: AsyncClient) -> None:
    response = await client.patch(_NOVA3, json=_PATCH, headers=_internal())
    assert response.status_code == 200
    state = response.json()
    assert state["running"] is False
    assert state["shown"] is True
    assert state["updated_by"] == INTERNAL_EMAIL

    listing = await client.get("/v1/admin/models", headers=_internal())
    nova = next(
        m
        for m in listing.json()["models"]
        if (m["benchmark"], m["provider"], m["model"]) == ("STT", "deepgram", "nova-3")
    )
    assert nova["state"]["running"] is False
    latest = nova["history"][0]
    assert (latest["old_running"], latest["old_shown"]) == (True, True)
    assert (latest["new_running"], latest["new_shown"]) == (False, True)
    assert latest["changed_by"] == INTERNAL_EMAIL


async def test_patch_unknown_model_is_404(client: AsyncClient) -> None:
    response = await client.patch(
        "/v1/admin/models/STT/deepgram/no-such-model", json=_PATCH, headers=_internal()
    )
    assert response.status_code == 404


async def test_patch_invalid_benchmark_is_422(client: AsyncClient) -> None:
    response = await client.patch(
        "/v1/admin/models/XTT/deepgram/nova-3", json=_PATCH, headers=_internal()
    )
    assert response.status_code == 422


async def test_toggle_reaches_public_endpoints_immediately(client: AsyncClient) -> None:
    """Hiding a model lands on /v1/providers at once (the PATCH busts the cache)."""

    def nova_models(payload: dict) -> set[str]:  # type: ignore[type-arg]
        entry = next((e for e in payload["stt"] if e["provider"] == "deepgram"), None)
        return {m["model"] for m in entry["models"]} if entry else set()

    before = (await client.get("/v1/providers")).json()
    assert "nova-3" in nova_models(before)

    hide = await client.patch(_NOVA3, json={"running": True, "shown": False}, headers=_internal())
    assert hide.status_code == 200

    after = (await client.get("/v1/providers")).json()
    assert "nova-3" not in nova_models(after)

    # A granted caller still sees it, flagged as embargoed.
    internal_view = (await client.get("/v1/providers", headers=_internal())).json()
    assert "nova-3" in nova_models(internal_view)


async def test_admin_responses_are_never_cached_shared(client: AsyncClient) -> None:
    response = await client.get("/v1/admin/models", headers=_internal())
    assert response.headers.get("Cache-Control") == "private, no-store"
