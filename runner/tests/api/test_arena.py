# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Voice Arena endpoints (GET/POST /v1/arena/*)."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.arena.pairing import active_tts_models
from coval_bench.providers.base import TTSResult
from tests.api.conftest import ARENA_LABELER_KEY, _make_db_url

_LABELER_HEADERS = {"X-Labeler-Key": ARENA_LABELER_KEY}


async def _apply_arena_schema(dsn: str) -> None:
    """Create the arena tables read by the endpoints (mirrors migration 20260615_0007)."""
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        await aconn.execute("CREATE SCHEMA IF NOT EXISTS arena")
        await aconn.execute("""
            CREATE TABLE IF NOT EXISTS arena.battles (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                provider_a  TEXT NOT NULL,
                model_a     TEXT NOT NULL,
                provider_b  TEXT NOT NULL,
                model_b     TEXT NOT NULL,
                domain      TEXT,
                prompt_text TEXT NOT NULL,
                audio_a_url TEXT NOT NULL,
                audio_b_url TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await aconn.execute("""
            CREATE TABLE IF NOT EXISTS arena.votes (
                id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                battle_id  UUID NOT NULL REFERENCES arena.battles(id),
                outcome    TEXT NOT NULL CHECK (outcome IN ('A_WIN','B_WIN','TIE')),
                voter_type TEXT NOT NULL CHECK (voter_type IN ('labeler','external')),
                voter_id   TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (battle_id, voter_type, voter_id)
            )
        """)
        await aconn.execute("""
            CREATE OR REPLACE FUNCTION arena.set_updated_at() RETURNS trigger AS $$
            BEGIN
                NEW.updated_at := now();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        await aconn.execute("DROP TRIGGER IF EXISTS votes_set_updated_at ON arena.votes")
        await aconn.execute("""
            CREATE TRIGGER votes_set_updated_at
                BEFORE UPDATE ON arena.votes
                FOR EACH ROW
                EXECUTE FUNCTION arena.set_updated_at()
        """)
        await aconn.execute("""
            CREATE TABLE IF NOT EXISTS arena.leaderboard_snapshots (
                id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                computed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                metric_name         TEXT NOT NULL,
                methodology_version TEXT NOT NULL,
                domain              TEXT NOT NULL DEFAULT 'all',
                provider            TEXT NOT NULL,
                model               TEXT NOT NULL,
                rating_elo          NUMERIC NOT NULL,
                rating_bt           NUMERIC NOT NULL,
                ci_low              NUMERIC,
                ci_high             NUMERIC,
                ci_half_width       NUMERIC,
                votes_total         INTEGER NOT NULL,
                wins                NUMERIC NOT NULL,
                losses              NUMERIC NOT NULL,
                ties                NUMERIC NOT NULL,
                status              TEXT NOT NULL
            )
        """)
    finally:
        await aconn.close()


async def _insert_battle(postgresql: Any, **kwargs: Any) -> str:
    """Insert a battle row and return its id as a string."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        defaults: dict[str, Any] = {
            "provider_a": "elevenlabs",
            "model_a": "eleven_multilingual_v2",
            "provider_b": "cartesia",
            "model_b": "sonic-3",
            "domain": "support",
            "prompt_text": "Tell me about your refund policy.",
            "audio_a_url": "https://example.test/a.wav",
            "audio_b_url": "https://example.test/b.wav",
        }
        defaults.update(kwargs)
        row = await aconn.execute(
            """
            INSERT INTO arena.battles
                (provider_a, model_a, provider_b, model_b, domain,
                 prompt_text, audio_a_url, audio_b_url)
            VALUES
                (%(provider_a)s, %(model_a)s, %(provider_b)s, %(model_b)s, %(domain)s,
                 %(prompt_text)s, %(audio_a_url)s, %(audio_b_url)s)
            RETURNING id
            """,
            defaults,
        )
        result = await row.fetchone()
        assert result is not None
        return str(result[0])
    finally:
        await aconn.close()


async def _insert_snapshot(postgresql: Any, **kwargs: Any) -> None:
    """Insert a leaderboard_snapshots row."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        defaults: dict[str, Any] = {
            "computed_at": datetime(2026, 6, 18, 12, 0, tzinfo=UTC),
            "metric_name": "naturalness",
            "methodology_version": "davidson-v1",
            "domain": "all",
            "provider": "elevenlabs",
            "model": "eleven_multilingual_v2",
            "rating_elo": 1500.0,
            "rating_bt": 0.0,
            "ci_low": 1450.0,
            "ci_high": 1550.0,
            "ci_half_width": 50.0,
            "votes_total": 10,
            "wins": 6,
            "losses": 4,
            "ties": 0,
            "status": "usable",
        }
        defaults.update(kwargs)
        await aconn.execute(
            """
            INSERT INTO arena.leaderboard_snapshots
                (computed_at, metric_name, methodology_version, domain, provider, model,
                 rating_elo, rating_bt, ci_low, ci_high, ci_half_width,
                 votes_total, wins, losses, ties, status)
            VALUES
                (%(computed_at)s, %(metric_name)s, %(methodology_version)s, %(domain)s,
                 %(provider)s, %(model)s, %(rating_elo)s, %(rating_bt)s, %(ci_low)s,
                 %(ci_high)s, %(ci_half_width)s, %(votes_total)s, %(wins)s, %(losses)s,
                 %(ties)s, %(status)s)
            """,
            defaults,
        )
    finally:
        await aconn.close()


async def test_get_battle_404_when_empty(client: AsyncClient, postgresql: Any) -> None:
    """No battles seeded -> 404."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get("/v1/arena/battle", headers=_LABELER_HEADERS)
    assert response.status_code == 404


async def test_get_battle_is_blind(client: AsyncClient, postgresql: Any) -> None:
    """A served battle exposes only the blind fields, never provider/model identities."""
    await _apply_arena_schema(_make_db_url(postgresql))
    await _insert_battle(postgresql)

    response = await client.get("/v1/arena/battle", headers=_LABELER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"id", "prompt_text", "domain", "audio_a_url", "audio_b_url"}
    assert "provider_a" not in data
    assert "model_a" not in data
    assert data["domain"] == "support"


async def test_get_battle_by_id(client: AsyncClient, postgresql: Any) -> None:
    """GET /arena/battle/{id} returns the matching battle."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)

    response = await client.get(f"/v1/arena/battle/{battle_id}", headers=_LABELER_HEADERS)
    assert response.status_code == 200
    assert response.json()["id"] == battle_id


async def test_get_battle_by_id_unknown_returns_404(client: AsyncClient, postgresql: Any) -> None:
    """A well-formed but unknown UUID returns 404."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get(
        "/v1/arena/battle/00000000-0000-0000-0000-000000000000", headers=_LABELER_HEADERS
    )
    assert response.status_code == 404


async def test_get_battle_by_id_malformed_returns_422(client: AsyncClient, postgresql: Any) -> None:
    """A non-UUID id is rejected by validation with 422."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get("/v1/arena/battle/not-a-uuid", headers=_LABELER_HEADERS)
    assert response.status_code == 422


async def test_leaderboard_empty_when_no_snapshots(client: AsyncClient, postgresql: Any) -> None:
    """No snapshots -> 200 with empty entries and null computed_at."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get("/v1/arena/leaderboard", headers=_LABELER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "naturalness"
    assert data["domain"] == "all"
    assert data["entries"] == []
    assert data["computed_at"] is None


async def test_leaderboard_returns_latest_board_sorted(
    client: AsyncClient, postgresql: Any
) -> None:
    """Only the most recent computed_at board is returned, sorted by rating_elo desc."""
    await _apply_arena_schema(_make_db_url(postgresql))
    stale = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    latest = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)

    # An older board that must be excluded.
    await _insert_snapshot(postgresql, computed_at=stale, provider="old", model="m", rating_elo=999)
    # The latest board: two models, inserted out of rank order.
    await _insert_snapshot(
        postgresql, computed_at=latest, provider="cartesia", model="sonic-3", rating_elo=1480
    )
    await _insert_snapshot(
        postgresql, computed_at=latest, provider="elevenlabs", model="v2", rating_elo=1520
    )

    response = await client.get("/v1/arena/leaderboard", headers=_LABELER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    entries = data["entries"]
    assert len(entries) == 2
    assert [e["model"] for e in entries] == ["v2", "sonic-3"]
    assert data["methodology_version"] == "davidson-v1"


async def test_leaderboard_domain_filter(client: AsyncClient, postgresql: Any) -> None:
    """domain filter returns only that domain's board, excluding the global 'all' board."""
    await _apply_arena_schema(_make_db_url(postgresql))
    computed = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)
    await _insert_snapshot(
        postgresql, computed_at=computed, domain="all", provider="elevenlabs", model="v2"
    )
    await _insert_snapshot(
        postgresql, computed_at=computed, domain="support", provider="cartesia", model="sonic-3"
    )

    support = await client.get(
        "/v1/arena/leaderboard", params={"domain": "support"}, headers=_LABELER_HEADERS
    )
    assert support.status_code == 200
    assert support.json()["domain"] == "support"
    assert [e["model"] for e in support.json()["entries"]] == ["sonic-3"]

    default = await client.get("/v1/arena/leaderboard", headers=_LABELER_HEADERS)
    assert [e["model"] for e in default.json()["entries"]] == ["v2"]


async def test_leaderboard_latest_is_scoped_per_metric(
    client: AsyncClient, postgresql: Any
) -> None:
    """The latest board is per metric: a metric computed earlier still returns its own rows."""
    await _apply_arena_schema(_make_db_url(postgresql))
    await _insert_snapshot(
        postgresql,
        computed_at=datetime(2026, 6, 18, 12, 0, tzinfo=UTC),
        metric_name="naturalness",
        provider="elevenlabs",
        model="v2",
    )
    await _insert_snapshot(
        postgresql,
        computed_at=datetime(2026, 6, 17, 12, 0, tzinfo=UTC),
        metric_name="clarity",
        provider="cartesia",
        model="sonic-3",
    )

    response = await client.get(
        "/v1/arena/leaderboard", params={"metric": "clarity"}, headers=_LABELER_HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "clarity"
    assert [e["model"] for e in data["entries"]] == ["sonic-3"]


async def test_leaderboard_does_not_mix_methodology_versions(
    client: AsyncClient, postgresql: Any
) -> None:
    """Two methodology versions sharing computed_at must not merge into one board."""
    await _apply_arena_schema(_make_db_url(postgresql))
    shared = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)
    await _insert_snapshot(
        postgresql,
        computed_at=shared,
        methodology_version="davidson-v1",
        provider="elevenlabs",
        model="v2",
    )
    await _insert_snapshot(
        postgresql,
        computed_at=shared,
        methodology_version="davidson-v2",
        provider="cartesia",
        model="sonic-3",
    )

    response = await client.get("/v1/arena/leaderboard", headers=_LABELER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    # One board only: the tiebreaker picks davidson-v2, so v1's row is excluded.
    assert data["methodology_version"] == "davidson-v2"
    assert [e["model"] for e in data["entries"]] == ["sonic-3"]


async def test_locked_reads_are_404_without_key(client: AsyncClient, postgresql: Any) -> None:
    """Without a labeler key the reads/generate return 404 — indistinguishable from a route
    that does not exist. Data is present, proving the lock fires before any DB access."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    await _insert_snapshot(postgresql)

    assert (await client.get("/v1/arena/battle")).status_code == 404
    assert (await client.get(f"/v1/arena/battle/{battle_id}")).status_code == 404
    assert (await client.get("/v1/arena/leaderboard")).status_code == 404
    assert (await client.post("/v1/arena/battle", json={"prompt": "hi"})).status_code == 404


async def test_locked_reads_are_404_with_wrong_key(client: AsyncClient, postgresql: Any) -> None:
    """A non-matching key is treated like no key: 404, never 403 (no existence leak)."""
    await _apply_arena_schema(_make_db_url(postgresql))
    await _insert_snapshot(postgresql)
    wrong = {"X-Labeler-Key": "not-the-key"}
    assert (await client.get("/v1/arena/leaderboard", headers=wrong)).status_code == 404


async def _count_votes(postgresql: Any, battle_id: str) -> int:
    """Return the number of vote rows for a battle."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        row = await aconn.execute(
            "SELECT count(*) FROM arena.votes WHERE battle_id = %s", (battle_id,)
        )
        result = await row.fetchone()
        assert result is not None
        return int(result[0])
    finally:
        await aconn.close()


async def test_vote_without_labeler_key_is_403(client: AsyncClient, postgresql: Any) -> None:
    """No labeler key -> 403 (external voting is not enabled)."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    response = await client.post(
        "/v1/arena/vote",
        json={"battle_id": battle_id, "outcome": "A_WIN", "voter_id": "ann-1"},
    )
    assert response.status_code == 403
    assert await _count_votes(postgresql, battle_id) == 0


async def test_vote_with_wrong_key_is_403(client: AsyncClient, postgresql: Any) -> None:
    """A non-matching labeler key -> 403."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    response = await client.post(
        "/v1/arena/vote",
        json={"battle_id": battle_id, "outcome": "A_WIN", "voter_id": "ann-1"},
        headers={"X-Labeler-Key": "not-the-key"},
    )
    assert response.status_code == 403


async def test_vote_records_a_labeler_vote(client: AsyncClient, postgresql: Any) -> None:
    """A valid labeler key persists the vote and stamps voter_type='labeler'."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    response = await client.post(
        "/v1/arena/vote",
        json={"battle_id": battle_id, "outcome": "A_WIN", "voter_id": "ann-1"},
        headers=_LABELER_HEADERS,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["battle_id"] == battle_id
    assert data["outcome"] == "A_WIN"
    assert data["voter_type"] == "labeler"
    assert data["voter_id"] == "ann-1"
    assert await _count_votes(postgresql, battle_id) == 1


async def test_vote_body_cannot_override_voter_type(client: AsyncClient, postgresql: Any) -> None:
    """A voter_type smuggled into the body is ignored; the server always pins 'labeler'."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    response = await client.post(
        "/v1/arena/vote",
        json={
            "battle_id": battle_id,
            "outcome": "A_WIN",
            "voter_id": "ann-1",
            "voter_type": "external",
        },
        headers=_LABELER_HEADERS,
    )
    assert response.status_code == 201
    assert response.json()["voter_type"] == "labeler"


async def test_revote_updates_existing_row(client: AsyncClient, postgresql: Any) -> None:
    """The same voter re-voting a battle updates their row, never adds a second."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    first = await client.post(
        "/v1/arena/vote",
        json={"battle_id": battle_id, "outcome": "A_WIN", "voter_id": "ann-1"},
        headers=_LABELER_HEADERS,
    )
    second = await client.post(
        "/v1/arena/vote",
        json={"battle_id": battle_id, "outcome": "B_WIN", "voter_id": "ann-1"},
        headers=_LABELER_HEADERS,
    )
    assert first.status_code == 201
    assert second.status_code == 201
    # Same row (dedup), outcome overwritten.
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["outcome"] == "B_WIN"
    assert await _count_votes(postgresql, battle_id) == 1


async def test_distinct_voters_create_separate_rows(client: AsyncClient, postgresql: Any) -> None:
    """Dedup is per-identity, not per-battle: two voter_ids on one battle keep both rows."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    for voter_id in ("ann-1", "ann-2"):
        response = await client.post(
            "/v1/arena/vote",
            json={"battle_id": battle_id, "outcome": "A_WIN", "voter_id": voter_id},
            headers=_LABELER_HEADERS,
        )
        assert response.status_code == 201
    assert await _count_votes(postgresql, battle_id) == 2


async def test_empty_voter_id_is_accepted(client: AsyncClient, postgresql: Any) -> None:
    """Pins current MVP behavior: voter_id is not validated, so an empty string is accepted."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    response = await client.post(
        "/v1/arena/vote",
        json={"battle_id": battle_id, "outcome": "A_WIN", "voter_id": ""},
        headers=_LABELER_HEADERS,
    )
    assert response.status_code == 201
    assert response.json()["voter_id"] == ""


async def test_vote_on_unknown_battle_is_404(client: AsyncClient, postgresql: Any) -> None:
    """A valid key but a battle id that does not exist -> 404."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.post(
        "/v1/arena/vote",
        json={
            "battle_id": "00000000-0000-0000-0000-000000000000",
            "outcome": "A_WIN",
            "voter_id": "ann-1",
        },
        headers=_LABELER_HEADERS,
    )
    assert response.status_code == 404


async def test_vote_with_invalid_outcome_is_422(client: AsyncClient, postgresql: Any) -> None:
    """An outcome outside the allowed set fails request validation with 422."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    response = await client.post(
        "/v1/arena/vote",
        json={"battle_id": battle_id, "outcome": "MAYBE", "voter_id": "ann-1"},
        headers=_LABELER_HEADERS,
    )
    assert response.status_code == 422


def _fake_tts_providers(fail_models: set[str]) -> dict[str, type]:
    """A fake TTS_PROVIDERS map: every roster provider returns a stub WAV path.

    Models named in *fail_models* return an error instead, to drive the failure path.
    """

    class _FakeTTS:
        def __init__(self, settings: Any, model: str, voice: str) -> None:
            self.model = model

        async def synthesize(self, text: str) -> TTSResult:
            if self.model in fail_models:
                return TTSResult(
                    provider="fake",
                    model=self.model,
                    voice="v",
                    ttfa_ms=None,
                    audio_path=None,
                    error="boom",
                )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                path = Path(handle.name)
            return TTSResult(
                provider="fake",
                model=self.model,
                voice="v",
                ttfa_ms=1.0,
                audio_path=path,
                error=None,
            )

    return {m.provider: _FakeTTS for m in active_tts_models()}


async def test_create_battle_returns_blind_battle(
    client: AsyncClient, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /arena/battle synthesizes a pair and returns a blind battle (201)."""
    await _apply_arena_schema(_make_db_url(postgresql))
    monkeypatch.setattr("coval_bench.arena.generate.TTS_PROVIDERS", _fake_tts_providers(set()))
    monkeypatch.setattr(
        "coval_bench.arena.generate.store_clip", lambda settings, src: f"/clips/{src.name}"
    )

    response = await client.post(
        "/v1/arena/battle",
        json={"prompt": "Your appointment is confirmed.", "domain": "customer service"},
        headers=_LABELER_HEADERS,
    )

    assert response.status_code == 201
    data = response.json()
    assert set(data) == {"id", "prompt_text", "domain", "audio_a_url", "audio_b_url"}
    assert "provider_a" not in data
    assert "model_a" not in data
    assert data["prompt_text"] == "Your appointment is confirmed."
    assert data["domain"] == "customer service"
    assert data["audio_a_url"].startswith("/clips/")


async def test_create_battle_rejects_empty_prompt(client: AsyncClient, postgresql: Any) -> None:
    """A blank prompt is rejected with 422 — no synthesis attempted."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.post(
        "/v1/arena/battle", json={"prompt": "   "}, headers=_LABELER_HEADERS
    )
    assert response.status_code == 422


async def test_create_battle_502_when_synthesis_fails(
    client: AsyncClient, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If synthesis fails, no battle is persisted and the endpoint returns 502."""
    await _apply_arena_schema(_make_db_url(postgresql))
    every_model = {m.model for m in active_tts_models()}
    monkeypatch.setattr(
        "coval_bench.arena.generate.TTS_PROVIDERS", _fake_tts_providers(every_model)
    )
    monkeypatch.setattr(
        "coval_bench.arena.generate.store_clip", lambda settings, src: "/clips/x.wav"
    )

    response = await client.post(
        "/v1/arena/battle", json={"prompt": "hello"}, headers=_LABELER_HEADERS
    )
    assert response.status_code == 502


async def test_reveal_without_labeler_key_is_403(client: AsyncClient, postgresql: Any) -> None:
    """No labeler key -> 403, identities never returned."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    response = await client.get(f"/v1/arena/battle/{battle_id}/reveal")
    assert response.status_code == 403


async def test_reveal_before_vote_is_409(client: AsyncClient, postgresql: Any) -> None:
    """A battle with no vote yet cannot be revealed."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    response = await client.get(f"/v1/arena/battle/{battle_id}/reveal", headers=_LABELER_HEADERS)
    assert response.status_code == 409


async def test_reveal_unknown_battle_is_404(client: AsyncClient, postgresql: Any) -> None:
    """A well-formed but unknown battle id returns 404."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get(
        "/v1/arena/battle/00000000-0000-0000-0000-000000000000/reveal",
        headers=_LABELER_HEADERS,
    )
    assert response.status_code == 404


async def test_reveal_after_vote_returns_identities(client: AsyncClient, postgresql: Any) -> None:
    """Once a vote exists, reveal returns both sides' provider/model."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)
    await client.post(
        "/v1/arena/vote",
        json={"battle_id": battle_id, "outcome": "A_WIN", "voter_id": "ann-1"},
        headers=_LABELER_HEADERS,
    )
    response = await client.get(f"/v1/arena/battle/{battle_id}/reveal", headers=_LABELER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["a"]["provider"] == "elevenlabs"
    assert data["a"]["model"] == "eleven_multilingual_v2"
    assert data["b"]["provider"] == "cartesia"
    assert data["b"]["model"] == "sonic-3"


async def test_create_battle_429_when_cap_reached(
    client: AsyncClient, app: FastAPI, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At/over the daily cap -> 429, and synthesis is never attempted."""
    await _apply_arena_schema(_make_db_url(postgresql))
    await _insert_battle(postgresql)  # one battle already today

    app.state.settings = app.state.settings.model_copy(update={"arena_daily_battle_cap": 1})

    async def _no_synth(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("generate_battle must not run once the cap is reached")

    monkeypatch.setattr("coval_bench.api.routers.arena.generate_battle", _no_synth)

    response = await client.post(
        "/v1/arena/battle", json={"prompt": "hello"}, headers=_LABELER_HEADERS
    )
    assert response.status_code == 429


async def test_create_battle_cap_disabled_when_zero(
    client: AsyncClient, app: FastAPI, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap of 0 disables the limit: generation proceeds even with battles present."""
    await _apply_arena_schema(_make_db_url(postgresql))
    await _insert_battle(postgresql)
    await _insert_battle(postgresql)

    app.state.settings = app.state.settings.model_copy(update={"arena_daily_battle_cap": 0})
    monkeypatch.setattr("coval_bench.arena.generate.TTS_PROVIDERS", _fake_tts_providers(set()))
    monkeypatch.setattr(
        "coval_bench.arena.generate.store_clip", lambda settings, src: f"/clips/{src.name}"
    )

    response = await client.post(
        "/v1/arena/battle", json={"prompt": "hello"}, headers=_LABELER_HEADERS
    )
    assert response.status_code == 201
