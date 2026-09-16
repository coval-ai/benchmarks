# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.db.registry_store.

Uses ``pytest-postgresql`` (embedded ``pg_ctl``, no Docker) to spin up a real
Postgres. No remote DB is ever contacted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db.registry_store import (
    DuplicateKey,
    NewModel,
    RegistryStore,
    StaleEdit,
    TagRecord,
)
from coval_bench.registries import Benchmark, Gender, Voice

from .conftest import apply_migrations, open_pool

registry_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB

_VOICES = (
    Voice(id="voice-f", gender=Gender.FEMALE, name="Ada"),
    Voice(id="voice-m", gender=Gender.MALE),
)


class _Identity(TypedDict):
    user_id: str
    org_id: str | None
    email: str | None


_EDITOR: _Identity = {"user_id": "user_editor", "org_id": None, "email": "editor@coval.dev"}


def _new_model(
    provider: str = "acme", model: str = "stt-1", modality: Benchmark = Benchmark.STT
) -> NewModel:
    return NewModel(modality=modality, provider=provider, model=model)


def _with_store(
    conn: psycopg.Connection[Any], scenario: Callable[[RegistryStore], Awaitable[None]]
) -> None:
    apply_migrations(conn)
    # The store's behaviour is what is under test here, not the seeded roster.
    with conn.transaction():
        conn.execute("DELETE FROM benchmarks_v2.model_history")
        conn.execute("DELETE FROM benchmarks_v2.models")
        conn.execute("DELETE FROM benchmarks_v2.tags")

    async def _run() -> None:
        pool = await open_pool(conn)
        try:
            await scenario(RegistryStore(pool))
        finally:
            await pool.close()

    asyncio.run(_run())


def _streaming_tag() -> TagRecord:
    return TagRecord(value="streaming", category="features", label="Streaming")


def test_tags_roundtrip_and_reject_duplicates(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        await store.insert_tag(TagRecord(value="vad", category="features", label="VAD"))
        await store.insert_tag(_streaming_tag())
        assert [tag.value for tag in await store.list_tags()] == ["streaming", "vad"]
        with pytest.raises(DuplicateKey):
            await store.insert_tag(_streaming_tag())

    _with_store(registry_pg, scenario)


def test_insert_model_roundtrips_and_writes_history(
    registry_pg: psycopg.Connection[Any],
) -> None:
    async def scenario(store: RegistryStore) -> None:
        await store.insert_tag(_streaming_tag())
        new = NewModel(
            modality=Benchmark.TTS,
            provider="acme",
            model="tts-1",
            voice="voice-f",
            voices=_VOICES,
            region="eu",
            tags=("streaming",),
        )
        created = await store.insert_model(new, **_EDITOR)

        fetched = await store.get_model(created.id)
        assert fetched == created
        assert await store.list_models() == [created]
        assert created.voices == _VOICES
        assert created.tags == ("streaming",)
        assert created.updated_by_user_id == "user_editor"
        assert created.updated_by_email == "editor@coval.dev"

        history = await store.recent_history()
        (change,) = history[created.id]
        assert change.old is None
        assert change.new["provider"] == "acme"
        assert "updated_by_email" not in change.new
        assert change.new["tags"] == ["streaming"]
        assert change.changed_by_user_id == "user_editor"
        assert change.changed_by_org_id is None

    _with_store(registry_pg, scenario)


def test_insert_rejects_a_duplicate_natural_key(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        await store.insert_model(_new_model(), **_EDITOR)
        with pytest.raises(DuplicateKey):
            await store.insert_model(_new_model(), **_EDITOR)
        # The same provider/model under another modality is a different model.
        await store.insert_model(_new_model(modality=Benchmark.TTS), **_EDITOR)

    _with_store(registry_pg, scenario)


def test_models_list_in_id_order(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        first = await store.insert_model(_new_model(model="stt-1"), **_EDITOR)
        second = await store.insert_model(_new_model(model="stt-2"), **_EDITOR)
        assert [record.id for record in await store.list_models()] == [first.id, second.id]
        assert await store.get_model(first.id + second.id + 1) is None

    _with_store(registry_pg, scenario)


def test_update_applies_changes_and_appends_history(
    registry_pg: psycopg.Connection[Any],
) -> None:
    async def scenario(store: RegistryStore) -> None:
        created = await store.insert_model(_new_model(), **_EDITOR)
        result = await store.update_model(
            created.id,
            {"published": True, "region": "us"},
            expected_updated_at=created.updated_at,
            user_id="user_second",
            org_id="org_partner",
            email=None,
        )
        assert result is not None
        before, after = result
        assert before == created
        assert (after.published, after.region) == (True, "us")
        assert after.updated_at != before.updated_at
        assert after.updated_by_user_id == "user_second"
        assert after.updated_by_email is None
        assert await store.get_model(created.id) == after

        changes = (await store.recent_history())[created.id]
        assert [change.old is None for change in changes] == [False, True]
        assert changes[0].old is not None
        assert changes[0].old["published"] is False
        assert changes[0].new["published"] is True
        assert changes[0].changed_by_org_id == "org_partner"

    _with_store(registry_pg, scenario)


def test_update_from_a_stale_row_raises(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        created = await store.insert_model(_new_model(), **_EDITOR)
        result = await store.update_model(
            created.id,
            {"published": True},
            expected_updated_at=created.updated_at,
            **_EDITOR,
        )
        assert result is not None
        _, after = result
        with pytest.raises(StaleEdit) as exc:
            await store.update_model(
                created.id,
                {"collected": False},
                expected_updated_at=created.updated_at,
                **_EDITOR,
            )
        assert exc.value.current == after
        assert len((await store.recent_history())[created.id]) == 2

    _with_store(registry_pg, scenario)


def test_a_noop_update_writes_nothing(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        created = await store.insert_model(_new_model(), **_EDITOR)
        result = await store.update_model(
            created.id,
            {"published": created.published, "provider": created.provider},
            expected_updated_at=created.updated_at,
            user_id="user_second",
            org_id=None,
            email=None,
        )
        assert result == (created, created)
        assert await store.get_model(created.id) == created
        assert len((await store.recent_history())[created.id]) == 1

    _with_store(registry_pg, scenario)


def test_update_replaces_the_tag_set(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        await store.insert_tag(_streaming_tag())
        await store.insert_tag(TagRecord(value="vad", category="features", label="VAD"))
        created = await store.insert_model(_new_model(), **_EDITOR)
        result = await store.update_model(
            created.id,
            {"tags": ("streaming", "vad")},
            expected_updated_at=created.updated_at,
            **_EDITOR,
        )
        assert result is not None
        assert result[1].tags == ("streaming", "vad")

        second = await store.update_model(
            created.id,
            {"tags": ("vad",)},
            expected_updated_at=result[1].updated_at,
            **_EDITOR,
        )
        assert second is not None
        assert second[1].tags == ("vad",)

    _with_store(registry_pg, scenario)


def test_tags_are_canonicalized(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        await store.insert_tag(_streaming_tag())
        await store.insert_tag(TagRecord(value="vad", category="features", label="VAD"))
        new = NewModel(
            modality=Benchmark.STT, provider="acme", model="stt-1", tags=("vad", "streaming")
        )
        created = await store.insert_model(new, **_EDITOR)
        assert created.tags == ("streaming", "vad")
        assert await store.get_model(created.id) == created

        result = await store.update_model(
            created.id,
            {"tags": ("vad", "streaming", "vad")},
            expected_updated_at=created.updated_at,
            **_EDITOR,
        )
        assert result == (created, created)
        assert len((await store.recent_history())[created.id]) == 1

    _with_store(registry_pg, scenario)


def test_a_rename_onto_an_existing_key_raises(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        await store.insert_model(_new_model(model="stt-1"), **_EDITOR)
        other = await store.insert_model(_new_model(model="stt-2"), **_EDITOR)
        with pytest.raises(DuplicateKey):
            await store.update_model(
                other.id,
                {"model": "stt-1"},
                expected_updated_at=other.updated_at,
                **_EDITOR,
            )

    _with_store(registry_pg, scenario)


def test_update_of_a_missing_model_returns_none(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        created = await store.insert_model(_new_model(), **_EDITOR)
        assert (
            await store.update_model(
                created.id + 1,
                {"published": True},
                expected_updated_at=created.updated_at,
                **_EDITOR,
            )
            is None
        )

    _with_store(registry_pg, scenario)


def test_the_modality_is_not_editable(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        created = await store.insert_model(_new_model(), **_EDITOR)
        with pytest.raises(ValueError, match="uneditable"):
            await store.update_model(
                created.id,
                {"modality": Benchmark.TTS},
                expected_updated_at=created.updated_at,
                **_EDITOR,
            )

    _with_store(registry_pg, scenario)


def test_color_roundtrips_and_clears(registry_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: RegistryStore) -> None:
        created = await store.insert_model(_new_model(), **_EDITOR)
        assert created.color is None
        result = await store.update_model(
            created.id, {"color": "#1db098"}, expected_updated_at=created.updated_at, **_EDITOR
        )
        assert result is not None
        _, painted = result
        assert painted.color == "#1db098"
        assert (await store.get_model(created.id)) == painted
        result = await store.update_model(
            created.id, {"color": None}, expected_updated_at=painted.updated_at, **_EDITOR
        )
        assert result is not None
        _, cleared = result
        assert cleared.color is None
        changes = await store.history(created.id)
        assert [change.new["color"] for change in changes] == [None, "#1db098", None]

    _with_store(registry_pg, scenario)


def test_llm_migration_preserves_existing_models_without_registering_variants(
    registry_pg: psycopg.Connection[Any],
) -> None:
    from pathlib import Path

    from alembic import command
    from alembic.config import Config
    from pydantic import SecretStr

    from coval_bench.db.registry_store import fetch_models
    from coval_bench.llm.coval_agent import CovalTextAgentDefinition

    from .conftest import async_dsn

    cfg = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    cfg.set_main_option(
        "sqlalchemy.url", async_dsn(registry_pg).replace("postgresql://", "postgresql+psycopg://")
    )
    command.upgrade(cfg, "20260915_0036")
    registry_pg.execute("""
        INSERT INTO benchmarks_v2.models
            (modality,provider,model,collected,published,updated_by_user_id)
        VALUES ('LLM','openai','gpt-4.1',true,false,'test'),
               ('LLM','google','gemini-2.5-flash',true,false,'test')
        ON CONFLICT DO NOTHING
    """)
    registry_pg.commit()
    command.upgrade(cfg, "head")

    async def verify() -> None:
        pool = await open_pool(registry_pg)
        try:
            models = {m.model: m for m in await fetch_models(pool) if m.benchmark is Benchmark.LLM}
            identities = set()
            assert set(models) == {"phonely-agent", "gpt-4.1", "gemini-2.5-flash"}
            for model in models.values():
                assert model.llm_config is not None
                definition = CovalTextAgentDefinition(
                    provider=model.provider,
                    model=model.model,
                    legacy_provider_route=model.llm_config.legacy_provider_route,
                    proxy_url="https://example.com",
                    proxy_secret=SecretStr("test"),
                    test_set_id="test",
                    instruction_metric_id="test",
                    persona_id="test",
                )
                identities.add(definition.customer_agent_id)
                if model.llm_config.legacy_provider_route:
                    assert definition.customer_agent_id == f"benchmarks-{model.provider}-text"
            assert len(identities) == len(models)
            google = models["gemini-2.5-flash"].llm_config
            assert google is not None and google.reasoning_effort == "none"
            history = await RegistryStore(pool).recent_history()
            assert any(
                change.old is not None
                and change.new.get("llm_config", {}).get("upstream_model") == "gpt-4.1"
                for changes in history.values()
                for change in changes
            )
        finally:
            await pool.close()

    asyncio.run(verify())
