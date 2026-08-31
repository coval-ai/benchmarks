# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The admin registry endpoints over HTTP."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from httpx import AsyncClient
from psycopg.types.json import Jsonb

from coval_bench.api.internal import _RETIRED_BOARD_KEYS
from coval_bench.api.routers import admin_models
from coval_bench.api.routers.admin_models import _implemented_providers as _real_implemented
from coval_bench.registries import Benchmark
from coval_bench.registries.metrics import METRIC_EXCLUSIONS
from tests.api.conftest import COVAL_ORG, EA_ORG, _make_db_url, bearer

_EXCLUDED_METRIC, _EXCLUDED_PAIRS = next(iter(METRIC_EXCLUSIONS.items()))
_EXCLUDED_PROVIDER, _EXCLUDED_MODEL = sorted(_EXCLUDED_PAIRS)[0]
_RETIRED_KEY, _CURRENT_KEY = next(iter(_RETIRED_BOARD_KEYS.items()))


@pytest.fixture(autouse=True)
def _empty_registry(postgresql: Any) -> None:
    """These tests own the registry tables; the seeded roster is not theirs."""
    conn = psycopg.connect(_make_db_url(postgresql))
    try:
        conn.execute("DELETE FROM benchmarks_v2.model_history")
        conn.execute("DELETE FROM benchmarks_v2.models")
        conn.execute("DELETE FROM benchmarks_v2.tags")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _fake_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the SDK-heavy provider imports; the real resolver has its own test."""
    implemented = frozenset({"acme", _EXCLUDED_PROVIDER})
    monkeypatch.setattr(
        admin_models,
        "_implemented_providers",
        lambda modality: None if modality is Benchmark.S2S else implemented,
    )


_VOICES_JSON = [
    {"id": "voice-f", "gender": "female", "name": "Ada", "accent": None},
    {"id": "voice-m", "gender": "male", "name": None, "accent": None},
]


def _admin_headers() -> dict[str, str]:
    return bearer(sub="user_admin", org_id=COVAL_ORG, email="admin@coval.dev")


async def _exec(postgresql: Any, sql: str, params: tuple[Any, ...]) -> Any:
    aconn = await psycopg.AsyncConnection.connect(_make_db_url(postgresql), autocommit=True)
    try:
        cur = await aconn.execute(sql, params)
        return (await cur.fetchone()) if cur.description else None
    finally:
        await aconn.close()


async def _seed_tag(postgresql: Any, value: str, category: str) -> None:
    await _exec(
        postgresql,
        "INSERT INTO benchmarks_v2.tags (value, category, label) VALUES (%s, %s, %s)",
        (value, category, value.capitalize()),
    )


async def _seed_model(postgresql: Any, **overrides: Any) -> int:
    fields: dict[str, Any] = {
        "modality": "STT",
        "provider": "acme",
        "model": "stt-1",
        "voice": None,
        "voices": Jsonb([]),
        "region": None,
        "collected": True,
        "published": False,
        "tags": (),
    }
    fields.update(overrides)
    tags = fields.pop("tags")
    row = await _exec(
        postgresql,
        """
        INSERT INTO benchmarks_v2.models
            (modality, provider, model, voice, voices, region, collected, published,
             updated_by_user_id, updated_by_email)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'user_seed', 'seed@coval.dev')
        RETURNING id
        """,
        (
            fields["modality"],
            fields["provider"],
            fields["model"],
            fields["voice"],
            fields["voices"],
            fields["region"],
            fields["collected"],
            fields["published"],
        ),
    )
    model_id = int(row[0])
    for tag in tags:
        await _exec(
            postgresql,
            "INSERT INTO benchmarks_v2.model_tags (model_id, tag) VALUES (%s, %s)",
            (model_id, tag),
        )
    return model_id


async def _seed_history(
    postgresql: Any, model_id: int, *, old: dict[str, Any] | None, new: dict[str, Any]
) -> None:
    await _exec(
        postgresql,
        """
        INSERT INTO benchmarks_v2.model_history
            (model_id, modality, provider, model, old, new, changed_by_user_id)
        VALUES (%s, 'STT', 'acme', 'stt-1', %s, %s, 'user_seed')
        """,
        (model_id, None if old is None else Jsonb(old), Jsonb(new)),
    )


async def test_admin_reads_require_a_token(client: AsyncClient) -> None:
    response = await client.get("/v1/admin/models")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_admin_reads_reject_partner_orgs(client: AsyncClient) -> None:
    headers = bearer(sub="user_partner", org_id=EA_ORG)
    assert (await client.get("/v1/admin/models", headers=headers)).status_code == 403


async def test_empty_registry_reads_as_empty(client: AsyncClient) -> None:
    response = await client.get("/v1/admin/models", headers=_admin_headers())
    assert response.status_code == 200
    assert response.json() == {"models": []}
    assert response.headers["Cache-Control"] == "private, no-store"
    assert "Authorization" in response.headers["Vary"]

    response = await client.get("/v1/tags")
    assert response.status_code == 200
    assert response.json() == {"tags": []}


async def test_models_list_state_tags_and_history(client: AsyncClient, postgresql: Any) -> None:
    await _seed_tag(postgresql, "streaming", "features")
    await _seed_tag(postgresql, "vad", "features")
    first = await _seed_model(
        postgresql,
        modality="TTS",
        model="tts-1",
        voice="voice-f",
        voices=Jsonb(_VOICES_JSON),
        region="eu",
        tags=("vad", "streaming"),
    )
    second = await _seed_model(postgresql, model="stt-2", collected=False)
    await _seed_history(postgresql, first, old=None, new={"published": False})
    await _seed_history(postgresql, first, old={"published": False}, new={"published": True})

    response = await client.get("/v1/admin/models", headers=_admin_headers())
    assert response.status_code == 200
    models = response.json()["models"]
    assert [entry["id"] for entry in models] == [first, second]

    tts = models[0]
    assert (tts["modality"], tts["model"], tts["voice"]) == ("TTS", "tts-1", "voice-f")
    assert (tts["source"], tts["licensing"], tts["region"]) == ("official-api", "proprietary", "eu")
    assert (tts["collected"], tts["published"], tts["arena_enabled"]) == (True, False, True)
    assert tts["voices"] == _VOICES_JSON
    assert tts["tags"] == ["streaming", "vad"]
    assert tts["updated_by_user_id"] == "user_seed"

    newest, oldest = tts["history"]
    assert (newest["old"], newest["new"]) == ({"published": False}, {"published": True})
    assert (oldest["old"], oldest["new"]) == (None, {"published": False})
    assert newest["changed_by_user_id"] == "user_seed"
    assert newest["changed_by_org_id"] is None

    assert models[1]["history"] == []
    assert models[1]["collected"] is False


async def test_the_tag_vocabulary_is_public(client: AsyncClient, postgresql: Any) -> None:
    await _seed_tag(postgresql, "vad", "features")
    await _seed_tag(postgresql, "streaming", "features")

    response = await client.get("/v1/tags")
    assert response.status_code == 200
    assert response.json() == {
        "tags": [
            {"value": "streaming", "category": "features", "label": "Streaming"},
            {"value": "vad", "category": "features", "label": "Vad"},
        ]
    }


def _create_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"modality": "STT", "provider": "acme", "model": "stt-new"}
    body.update(overrides)
    return body


async def _post_model(client: AsyncClient, **overrides: Any) -> Any:
    return await client.post(
        "/v1/admin/models", json=_create_body(**overrides), headers=_admin_headers()
    )


def test_the_real_provider_registries_resolve() -> None:
    stt = _real_implemented(Benchmark.STT)
    tts = _real_implemented(Benchmark.TTS)
    assert stt is not None and "deepgram" in stt
    # Optional-SDK providers resolve even where their extras are not installed.
    assert tts is not None and "google" in tts
    assert _real_implemented(Benchmark.S2S) is None


async def test_create_defaults_to_hidden(client: AsyncClient) -> None:
    response = await _post_model(client)
    assert response.status_code == 201
    created = response.json()
    assert (created["collected"], created["published"]) == (True, False)
    assert (created["source"], created["licensing"]) == ("official-api", "proprietary")
    assert created["updated_by_user_id"] == "user_admin"
    assert created["updated_by_email"] == "admin@coval.dev"

    (change,) = created["history"]  # the creation entry rides the POST response
    assert change["old"] is None
    assert change["new"]["model"] == "stt-new"

    listed = (await client.get("/v1/admin/models", headers=_admin_headers())).json()["models"]
    assert [entry["id"] for entry in listed] == [created["id"]]
    (change,) = listed[0]["history"]
    assert change["old"] is None
    assert change["new"]["model"] == "stt-new"
    assert change["changed_by_user_id"] == "user_admin"
    assert change["changed_by_org_id"] is None


async def test_create_requires_a_coval_token(client: AsyncClient) -> None:
    assert (await client.post("/v1/admin/models", json=_create_body())).status_code == 401
    partner = bearer(sub="user_partner", org_id=EA_ORG)
    response = await client.post("/v1/admin/models", json=_create_body(), headers=partner)
    assert response.status_code == 403


async def test_create_rejects_duplicates_and_bad_input(client: AsyncClient) -> None:
    assert (await _post_model(client)).status_code == 201
    assert (await _post_model(client)).status_code == 409
    assert (await _post_model(client, provider="nope")).status_code == 422
    assert (await _post_model(client, model="tagged", tags=["mystery"])).status_code == 422


async def test_a_stopped_model_may_name_a_dead_provider(client: AsyncClient) -> None:
    """History rows for providers whose code is gone stay importable, never collectable."""
    created = await _post_model(client, provider="gone", collected=False, published=False)
    assert created.status_code == 201
    body = created.json()
    revive = await _patch(client, body["id"], f'"{body["updated_at"]}"', {"collected": True})
    assert revive.status_code == 422


async def test_create_s2s_skips_the_provider_check(client: AsyncClient) -> None:
    response = await _post_model(client, modality="S2S", provider="anyone", model="s2s-1")
    assert response.status_code == 201


async def test_collected_tts_needs_a_voice(client: AsyncClient) -> None:
    assert (await _post_model(client, modality="TTS", model="tts-1")).status_code == 422
    ok = await _post_model(client, modality="TTS", model="tts-1", voice="v")
    assert ok.status_code == 201
    paused = await _post_model(client, modality="TTS", model="tts-2", collected=False)
    assert paused.status_code == 201


async def test_a_voice_pool_is_one_female_one_male(client: AsyncClient) -> None:
    lopsided = [{"id": "a", "gender": "female"}, {"id": "b", "gender": "female"}]
    response = await _post_model(client, modality="TTS", model="tts-1", voices=lopsided)
    assert response.status_code == 422
    shared_id = [{"id": "a", "gender": "female"}, {"id": "a", "gender": "male"}]
    response = await _post_model(client, modality="TTS", model="tts-1", voices=shared_id)
    assert response.status_code == 422
    paired = [{"id": "a", "gender": "female"}, {"id": "b", "gender": "male"}]
    response = await _post_model(client, modality="TTS", model="tts-1", voices=paired)
    assert response.status_code == 201


async def test_voices_are_tts_only(client: AsyncClient) -> None:
    assert (await _post_model(client, voice="v")).status_code == 422
    s2s = await _post_model(client, modality="S2S", provider="anyone", model="s2s-1", voice="v")
    assert s2s.status_code == 422
    pool = [{"id": "a", "gender": "female"}, {"id": "b", "gender": "male"}]
    assert (await _post_model(client, voices=pool)).status_code == 422


async def test_empty_strings_are_rejected_before_the_database(client: AsyncClient) -> None:
    assert (await _post_model(client, modality="TTS", voice="")).status_code == 422
    assert (await _post_model(client, creator="")).status_code == 422
    created = (await _post_model(client)).json()
    response = await _patch(client, created["id"], created["updated_at"], {"voice": ""})
    assert response.status_code == 422
    body = {"value": "", "category": "features", "label": "Streaming"}
    admin = _admin_headers()
    assert (await client.post("/v1/tags", json=body, headers=admin)).status_code == 422
    body = {"value": "streaming", "category": "features", "label": ""}
    assert (await client.post("/v1/tags", json=body, headers=admin)).status_code == 422


async def _patch(
    client: AsyncClient, model_id: int, stamp: str | None, body: dict[str, Any]
) -> Any:
    headers = _admin_headers()
    if stamp is not None:
        headers["If-Match"] = stamp
    return await client.patch(f"/v1/admin/models/{model_id}", json=body, headers=headers)


async def test_patch_flips_state_and_records_history(client: AsyncClient) -> None:
    created = (await _post_model(client)).json()
    response = await _patch(client, created["id"], created["updated_at"], {"published": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["warnings"] == []
    assert payload["model"]["published"] is True
    assert payload["model"]["updated_at"] != created["updated_at"]
    newest = payload["model"]["history"][0]
    assert (newest["old"]["published"], newest["new"]["published"]) == (False, True)


async def test_patch_requires_the_precondition_tag(client: AsyncClient) -> None:
    created = (await _post_model(client)).json()
    assert (await _patch(client, created["id"], None, {"published": True})).status_code == 428
    assert (await _patch(client, created["id"], "*", {})).status_code == 400
    # An unreadable or foreign tag matches nothing, same as a stale one.
    assert (await _patch(client, created["id"], "not-a-time", {})).status_code == 412
    assert (await _patch(client, created["id"], "2026-01-01T00:00:00", {})).status_code == 412


async def test_the_response_etag_round_trips(client: AsyncClient) -> None:
    response = await _post_model(client)
    etag = response.headers["ETag"]
    assert etag == f'"{response.json()["updated_at"]}"'
    patched = await _patch(client, response.json()["id"], etag, {"published": True})
    assert patched.status_code == 200
    assert patched.headers["ETag"] == f'"{patched.json()["model"]["updated_at"]}"'


async def test_patch_from_a_stale_stamp_is_412(client: AsyncClient) -> None:
    created = (await _post_model(client)).json()
    first = await _patch(client, created["id"], created["updated_at"], {"published": True})
    assert first.status_code == 200
    stale = await _patch(client, created["id"], created["updated_at"], {"collected": False})
    assert stale.status_code == 412


async def test_a_noop_patch_writes_no_history(client: AsyncClient) -> None:
    created = (await _post_model(client)).json()
    response = await _patch(client, created["id"], created["updated_at"], {"collected": True})
    assert response.status_code == 200
    assert len(response.json()["model"]["history"]) == 1  # the creation row only


async def test_patch_null_rules(client: AsyncClient) -> None:
    created = (await _post_model(client)).json()
    rejected = await _patch(client, created["id"], created["updated_at"], {"provider": None})
    assert rejected.status_code == 422
    allowed = await _patch(client, created["id"], created["updated_at"], {"creator": None})
    assert allowed.status_code == 200


async def test_patch_unknown_model_is_404(client: AsyncClient) -> None:
    created = (await _post_model(client)).json()
    response = await _patch(client, created["id"] + 1, created["updated_at"], {})
    assert response.status_code == 404


async def test_patch_rename_collision_is_409(client: AsyncClient) -> None:
    first = (await _post_model(client, model="stt-1")).json()
    second = (await _post_model(client, model="stt-2")).json()
    response = await _patch(client, second["id"], second["updated_at"], {"model": "stt-1"})
    assert response.status_code == 409
    assert first["model"] == "stt-1"


async def test_a_rename_warns_about_code_references(client: AsyncClient) -> None:
    created = (await _post_model(client, provider=_EXCLUDED_PROVIDER, model=_EXCLUDED_MODEL)).json()
    response = await _patch(
        client, created["id"], created["updated_at"], {"model": f"{_EXCLUDED_MODEL}-renamed"}
    )
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert any("METRIC_EXCLUSIONS" in warning for warning in warnings)


async def test_a_rename_warns_about_retired_board_keys(client: AsyncClient) -> None:
    provider, model = _CURRENT_KEY
    created = (await _post_model(client, modality="S2S", provider=provider, model=model)).json()
    response = await _patch(
        client, created["id"], created["updated_at"], {"model": _RETIRED_KEY[1]}
    )
    assert response.status_code == 200
    warnings = "\n".join(response.json()["warnings"])
    assert "no longer exists" in warnings
    if _RETIRED_KEY[0] == provider:
        assert "retired board key" in warnings


async def test_tag_creation_is_coval_gated(client: AsyncClient) -> None:
    body = {"value": "emotion-control", "category": "features", "label": "Emotion control"}
    assert (await client.post("/v1/tags", json=body)).status_code == 401
    partner = bearer(sub="user_partner", org_id=EA_ORG)
    assert (await client.post("/v1/tags", json=body, headers=partner)).status_code == 403

    created = await client.post("/v1/tags", json=body, headers=_admin_headers())
    assert created.status_code == 201
    assert created.json() == body
    assert (await client.post("/v1/tags", json=body, headers=_admin_headers())).status_code == 409
    assert (await client.get("/v1/tags")).json() == {"tags": [body]}
