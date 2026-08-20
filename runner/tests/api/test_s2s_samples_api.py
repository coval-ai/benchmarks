# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The S2S samples endpoints: who sees which recording, and how audio is served.

The bucket is private, so these routes are the only way in. A recording carries
its own transcript, so an embargoed model must vanish from the manifest entirely
rather than merely losing its audio link.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api.deps import get_settings
from coval_bench.config import Settings
from coval_bench.registries import MODEL_REGISTRY, Benchmark, RegisteredModel
from coval_bench.s2s.samples import AUDIO_URL_TTL
from tests.api.conftest import INTERNAL_EMAIL, _make_db_url, bearer

_BUCKET = "test-s2s-samples"
_SAMPLE = "2026-07-30T00:00:00Z"
_OTHER_SAMPLE = "2026-07-29T00:00:00Z"
_SIGNED = "https://storage.googleapis.com/signed-for-test"

_PARTNER_ORG = "org_s2s_partner"
_ORG_PROVIDERS = f'{{"{_PARTNER_ORG}": ["acme/secret-s2s"]}}'


def _internal() -> dict[str, str]:
    return bearer(email=INTERNAL_EMAIL)


def _partner() -> dict[str, str]:
    return bearer(org_id=_PARTNER_ORG)


_LIVE = ("openlab", "public-s2s")
_EMBARGOED = ("acme", "secret-s2s")


def _recording(provider: str, model: str, sample_id: str = _SAMPLE) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "object": f"s2s-samples/{sample_id}/{provider}/{model}.wav",
        "coval_run_id": "run-1",
        "sim_id": "sim-1",
        "agent_id": "agent-1",
        "turns": [{"index": 0, "role": "assistant", "content": f"{provider}-spoken-words"}],
    }


_OBJECTS: dict[str, Any] = {
    "s2s-samples/index.json": [_OTHER_SAMPLE, _SAMPLE],
    f"s2s-samples/{_SAMPLE}/manifest.json": {
        "schema_version": 2,
        "bucket_at": _SAMPLE,
        "test_case_id": "tc-1",
        "persona_name": "Standard Male",
        "recordings": [_recording(*_LIVE), _recording(*_EMBARGOED)],
    },
    # A tick whose only recording is embargoed: nothing about it may reach the public.
    f"s2s-samples/{_OTHER_SAMPLE}/manifest.json": {
        "schema_version": 2,
        "bucket_at": _OTHER_SAMPLE,
        "test_case_id": "tc-secret",
        "persona_name": "Standard Female",
        "recordings": [_recording(*_EMBARGOED, sample_id=_OTHER_SAMPLE)],
    },
}


@pytest.fixture(autouse=True)
def s2s_samples_env(postgresql: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A two-model S2S roster — one live, one embargoed — over a stubbed bucket.

    The live model gets an Active ``model_state`` row; the embargoed one gets
    none, taking the missing-row default (Hidden) that is the embargo itself.
    """
    patched = [
        *MODEL_REGISTRY,
        RegisteredModel(benchmark=Benchmark.S2S, provider=_LIVE[0], model=_LIVE[1]),
        RegisteredModel(benchmark=Benchmark.S2S, provider=_EMBARGOED[0], model=_EMBARGOED[1]),
    ]
    monkeypatch.setattr("coval_bench.api.internal.MODEL_REGISTRY", patched)
    monkeypatch.setattr("coval_bench.api.routers.s2s_samples.MODEL_REGISTRY", patched)
    monkeypatch.setattr("coval_bench.db.model_state.MODEL_REGISTRY", patched)
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO benchmarks_v2.model_state
                (benchmark, provider, model, running, shown, updated_by)
            VALUES ('S2S', %s, %s, true, true, 'test-setup')
            ON CONFLICT (benchmark, provider, model) DO UPDATE
                SET running = true, shown = true
            """,
            _LIVE,
        )

    def _read_json(bucket_name: str, key: str, **_: Any) -> Any:
        assert bucket_name == _BUCKET
        return _OBJECTS.get(key)

    monkeypatch.setattr("coval_bench.s2s.samples.read_json", _read_json)
    monkeypatch.setattr(
        "coval_bench.api.routers.s2s_samples.signed_url",
        lambda bucket_name, key, **_: f"{_SIGNED}/{key}",
    )


@pytest_asyncio.fixture
async def samples_client(app: FastAPI, client: AsyncClient) -> AsyncClient:
    """The API client with the samples bucket configured."""
    app.dependency_overrides[get_settings] = lambda: Settings(
        s2s_samples_bucket=_BUCKET,
        clerk_org_providers=_ORG_PROVIDERS,
    )
    return client


def _pairs(payload: dict[str, Any]) -> list[tuple[str, str]]:
    return [(r["provider"], r["model"]) for r in payload["recordings"]]


# --- the index --------------------------------------------------------------


async def test_index_lists_samples_newest_first(samples_client: AsyncClient) -> None:
    res = await samples_client.get("/v1/s2s/samples")

    assert res.status_code == 200
    assert res.json() == [_SAMPLE, _OTHER_SAMPLE]


async def test_index_is_never_shared_between_callers(samples_client: AsyncClient) -> None:
    res = await samples_client.get("/v1/s2s/samples")

    assert res.headers["cache-control"] == "private, no-store"
    assert "Authorization" in res.headers["vary"]


async def test_index_empty_when_no_bucket_is_configured(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(s2s_samples_bucket="")

    res = await client.get("/v1/s2s/samples")

    assert res.status_code == 200
    assert res.json() == []


# --- the manifest -----------------------------------------------------------


async def test_public_caller_never_sees_the_embargoed_recording(
    samples_client: AsyncClient,
) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}")

    assert res.status_code == 200
    assert _pairs(res.json()) == [_LIVE]


async def test_embargoed_transcript_leaves_with_its_recording(
    samples_client: AsyncClient,
) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}")

    assert f"{_EMBARGOED[0]}-spoken-words" not in res.text
    assert f"{_LIVE[0]}-spoken-words" in res.text


async def test_internal_caller_sees_every_recording(samples_client: AsyncClient) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}", headers=_internal())

    assert _pairs(res.json()) == [_LIVE, _EMBARGOED]


async def test_partner_token_unlocks_only_its_own_model(samples_client: AsyncClient) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}", headers=_partner())

    assert _pairs(res.json()) == [_LIVE, _EMBARGOED]
    assert res.headers["X-EA-Token-Status"] == "accepted"


async def test_unknown_token_falls_back_to_the_public_view(samples_client: AsyncClient) -> None:
    res = await samples_client.get(
        f"/v1/s2s/samples/{_SAMPLE}", headers={"Authorization": "Bearer not-a-real-token"}
    )

    assert _pairs(res.json()) == [_LIVE]
    assert res.headers["X-EA-Token-Status"] == "unknown"


async def test_manifest_hands_back_api_paths_not_storage_paths(
    samples_client: AsyncClient,
) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}", headers=_internal())
    body = res.json()

    assert body["sample_id"] == _SAMPLE
    for recording in body["recordings"]:
        assert recording["audio_path"].startswith("/v1/s2s/samples/")
        assert "object" not in recording
    assert "storage.googleapis.com" not in res.text


async def test_missing_sample_is_a_404(samples_client: AsyncClient) -> None:
    res = await samples_client.get("/v1/s2s/samples/2020-01-01T00:00:00Z")

    assert res.status_code == 404


async def test_malformed_sample_id_is_rejected_before_any_read(
    samples_client: AsyncClient,
) -> None:
    res = await samples_client.get("/v1/s2s/samples/banana")

    assert res.status_code == 422


async def test_a_wholly_embargoed_sample_is_a_404_not_an_empty_shell(
    samples_client: AsyncClient,
) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_OTHER_SAMPLE}")

    assert res.status_code == 404
    assert "tc-secret" not in res.text
    assert "Standard Female" not in res.text


async def test_the_same_sample_is_served_to_a_caller_who_may_hear_it(
    samples_client: AsyncClient,
) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_OTHER_SAMPLE}", headers=_partner())

    assert res.status_code == 200
    assert _pairs(res.json()) == [_EMBARGOED]


# --- the audio url ----------------------------------------------------------


async def test_audio_hands_back_a_signed_url_in_the_body(samples_client: AsyncClient) -> None:
    """A body, not a redirect: a browser cannot carry its token through one."""
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}/{_LIVE[0]}/{_LIVE[1]}/audio")

    assert res.status_code == 200
    assert res.json()["url"].startswith(_SIGNED)


async def test_audio_says_when_the_url_stops_working(samples_client: AsyncClient) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}/{_LIVE[0]}/{_LIVE[1]}/audio")

    expires_at = datetime.fromisoformat(res.json()["expires_at"])
    assert timedelta() < expires_at - datetime.now(UTC) <= AUDIO_URL_TTL


async def test_audio_url_is_never_cached(samples_client: AsyncClient) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}/{_LIVE[0]}/{_LIVE[1]}/audio")

    assert res.headers["cache-control"] == "private, no-store"
    assert "Authorization" in res.headers["vary"]
    assert res.headers["X-EA-Token-Status"] == "absent"


async def test_audio_for_an_embargoed_model_is_refused(samples_client: AsyncClient) -> None:
    res = await samples_client.get(
        f"/v1/s2s/samples/{_SAMPLE}/{_EMBARGOED[0]}/{_EMBARGOED[1]}/audio",
    )

    assert res.status_code == 404


async def test_audio_for_an_embargoed_model_is_served_to_internal(
    samples_client: AsyncClient,
) -> None:
    res = await samples_client.get(
        f"/v1/s2s/samples/{_SAMPLE}/{_EMBARGOED[0]}/{_EMBARGOED[1]}/audio",
        headers=_internal(),
    )

    assert res.status_code == 200
    assert res.headers["X-EA-Token-Status"] == "accepted"


async def test_audio_for_an_unknown_recording_is_a_404(samples_client: AsyncClient) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_SAMPLE}/{_LIVE[0]}/no-such-model/audio")

    assert res.status_code == 404


# --- errors are caller-scoped too -------------------------------------------


async def test_a_refusal_is_never_cached(samples_client: AsyncClient) -> None:
    """The public 404 and the partner's signed URL share a route, so no cache may reuse it."""
    res = await samples_client.get(
        f"/v1/s2s/samples/{_SAMPLE}/{_EMBARGOED[0]}/{_EMBARGOED[1]}/audio",
    )

    assert res.status_code == 404
    assert res.headers["cache-control"] == "private, no-store"
    assert "Authorization" in res.headers["vary"]


async def test_an_embargoed_manifest_404_is_never_cached(samples_client: AsyncClient) -> None:
    res = await samples_client.get(f"/v1/s2s/samples/{_OTHER_SAMPLE}")

    assert res.status_code == 404
    assert res.headers["cache-control"] == "private, no-store"
    assert "Authorization" in res.headers["vary"]


async def test_a_rejected_sample_id_is_never_cached(samples_client: AsyncClient) -> None:
    res = await samples_client.get("/v1/s2s/samples/banana")

    assert res.status_code == 422
    assert res.headers["cache-control"] == "private, no-store"
    assert "Authorization" in res.headers["vary"]
