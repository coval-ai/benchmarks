# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``RegistryStore`` — typed read/write helpers for the model registry tables.

Mirrors ``db/arena_store.py``: async, uses the shared psycopg pool, all SQL is
parameterised (``%s``). Validation of what a row may say (tag vocabulary,
voice pool rules, provider existence) lives at the API boundary; this layer is
mechanical and enforces only atomicity, natural-key uniqueness, and the
edit-from-stale-row check. Every write lands a full before/after snapshot in
``model_history``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

import psycopg
import psycopg.errors
import psycopg.rows
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, field_validator

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import Licensing, Source, Voice

RegistryPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

# Columns a PATCH may change. The modality is immutable: it selects the
# pipeline, so a modality flip is a new model, not an edit.
EDITABLE_FIELDS = frozenset(
    {
        "provider",
        "model",
        "voice",
        "voices",
        "creator",
        "source",
        "licensing",
        "on_prem",
        "region",
        "arena_enabled",
        "collected",
        "published",
        "tags",
    }
)


class DuplicateKey(Exception):
    """The write collided with an existing natural key or tag value."""


class StaleEdit(Exception):
    """The caller edited from a row that has since changed."""

    def __init__(self, current: ModelRecord) -> None:
        super().__init__(f"model {current.id} changed at {current.updated_at.isoformat()}")
        self.current = current


class TagRecord(BaseModel):
    """A row in ``benchmarks_v2.tags``."""

    value: str
    category: Literal["mode", "features"]
    label: str


class NewModel(BaseModel):
    """A model as POSTed, before the DB assigns identity and provenance."""

    modality: Benchmark
    provider: str
    model: str
    voice: str | None = None
    voices: tuple[Voice, ...] = ()
    creator: str | None = None
    source: Source = Source.OFFICIAL_API
    licensing: Licensing = Licensing.PROPRIETARY
    on_prem: bool = False
    region: Literal["us", "eu", "asia"] | None = None
    arena_enabled: bool = True
    collected: bool = True
    published: bool = False
    tags: tuple[str, ...] = ()

    @field_validator("tags")
    @classmethod
    def _canonical_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Tags are a set; reads alphabetize them, so writes must compare equal."""
        return tuple(sorted(set(value)))


class ModelRecord(NewModel):
    """A row in ``benchmarks_v2.models`` with its tags."""

    id: int
    updated_by_user_id: str
    updated_by_email: str | None = None
    updated_at: datetime


class ModelChange(BaseModel):
    """A row in ``benchmarks_v2.model_history``."""

    id: int
    model_id: int
    modality: Benchmark
    provider: str
    model: str
    old: dict[str, Any] | None = None
    new: dict[str, Any]
    changed_by_user_id: str
    changed_by_org_id: str | None = None
    changed_by_email: str | None = None
    changed_at: datetime


_SELECT_MODELS = """
    SELECT m.id, m.modality, m.provider, m.model, m.voice, m.voices, m.creator,
           m.source, m.licensing, m.on_prem, m.region, m.arena_enabled,
           m.collected, m.published, m.updated_by_user_id, m.updated_by_email,
           m.updated_at, COALESCE(t.tags, '{}') AS tags
    FROM benchmarks_v2.models m
    LEFT JOIN (
        SELECT model_id, array_agg(tag ORDER BY tag) AS tags
        FROM benchmarks_v2.model_tags
        GROUP BY model_id
    ) t ON t.model_id = m.id
"""

_INSERT_MODEL = """
    INSERT INTO benchmarks_v2.models
        (modality, provider, model, voice, voices, creator, source, licensing,
         on_prem, region, arena_enabled, collected, published,
         updated_by_user_id, updated_by_email)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id, updated_at
"""

_UPDATE_MODEL = """
    UPDATE benchmarks_v2.models
    SET provider = %s, model = %s, voice = %s, voices = %s, creator = %s,
        source = %s, licensing = %s, on_prem = %s, region = %s,
        arena_enabled = %s, collected = %s, published = %s,
        updated_by_user_id = %s, updated_by_email = %s,
        -- Strictly monotonic per row: the 412 stale check compares this exactly,
        -- so same-microsecond updates must still produce a new value.
        updated_at = GREATEST(now(), updated_at + interval '1 microsecond')
    WHERE id = %s
    RETURNING updated_at
"""

_INSERT_HISTORY = """
    INSERT INTO benchmarks_v2.model_history
        (model_id, modality, provider, model, old, new,
         changed_by_user_id, changed_by_org_id, changed_by_email)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _record(row: Mapping[str, Any]) -> ModelRecord:
    return ModelRecord.model_validate({**row, "tags": tuple(row["tags"])})


def _snapshot(record: ModelRecord) -> Jsonb:
    return Jsonb(record.model_dump(mode="json"))


class RegistryStore:
    """Per-pool persistence helper for the model and tag registries."""

    def __init__(self, pool: RegistryPool) -> None:
        self._pool = pool

    async def list_tags(self) -> list[TagRecord]:
        sql = "SELECT value, category, label FROM benchmarks_v2.tags ORDER BY value"
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql)).fetchall()
        return [TagRecord.model_validate(dict(row)) for row in rows]

    async def insert_tag(self, tag: TagRecord) -> TagRecord:
        sql = """
            INSERT INTO benchmarks_v2.tags (value, category, label)
            VALUES (%s, %s, %s)
        """
        try:
            async with self._pool.connection() as conn, conn.transaction():
                await conn.execute(sql, (tag.value, tag.category, tag.label))
        except psycopg.errors.UniqueViolation as exc:
            raise DuplicateKey(f"tag {tag.value!r} already exists") from exc
        return tag

    async def list_models(self) -> list[ModelRecord]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(f"{_SELECT_MODELS} ORDER BY m.id")).fetchall()
        return [_record(row) for row in rows]

    async def get_model(self, model_id: int) -> ModelRecord | None:
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(f"{_SELECT_MODELS} WHERE m.id = %s", (model_id,))
            ).fetchone()
        return None if row is None else _record(row)

    async def insert_model(
        self,
        new: NewModel,
        *,
        user_id: str,
        org_id: str | None,
        email: str | None,
    ) -> ModelRecord:
        """Insert a model with its tags and a creation history row, atomically."""
        try:
            async with self._pool.connection() as conn, conn.transaction():
                row = await (
                    await conn.execute(
                        _INSERT_MODEL,
                        (
                            new.modality,
                            new.provider,
                            new.model,
                            new.voice,
                            Jsonb([voice.model_dump(mode="json") for voice in new.voices]),
                            new.creator,
                            new.source,
                            new.licensing,
                            new.on_prem,
                            new.region,
                            new.arena_enabled,
                            new.collected,
                            new.published,
                            user_id,
                            email,
                        ),
                    )
                ).fetchone()
                if row is None:  # pragma: no cover — unreachable after INSERT RETURNING
                    raise RuntimeError("INSERT INTO benchmarks_v2.models returned no row")
                record = ModelRecord.model_validate(
                    {
                        **new.model_dump(),
                        "id": row["id"],
                        "updated_by_user_id": user_id,
                        "updated_by_email": email,
                        "updated_at": row["updated_at"],
                    }
                )
                await self._write_tags(conn, record.id, new.tags)
                await conn.execute(
                    _INSERT_HISTORY,
                    (
                        record.id,
                        record.modality,
                        record.provider,
                        record.model,
                        None,
                        _snapshot(record),
                        user_id,
                        org_id,
                        email,
                    ),
                )
        except psycopg.errors.UniqueViolation as exc:
            raise DuplicateKey(
                f"model ({new.modality}, {new.provider}, {new.model}) already exists"
            ) from exc
        return record

    async def update_model(
        self,
        model_id: int,
        changes: Mapping[str, Any],
        *,
        expected_updated_at: datetime,
        user_id: str,
        org_id: str | None,
        email: str | None,
    ) -> tuple[ModelRecord, ModelRecord] | None:
        """Apply *changes* to one model; return (before, after), or ``None`` if absent.

        Raises ``StaleEdit`` when the row changed since *expected_updated_at*, and
        ``DuplicateKey`` when a rename collides. A no-op returns before as after
        and writes nothing, so history only ever records real differences.
        """
        unknown = set(changes) - EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"uneditable fields: {sorted(unknown)}")
        try:
            async with self._pool.connection() as conn, conn.transaction():
                row = await (
                    await conn.execute(
                        f"{_SELECT_MODELS} WHERE m.id = %s FOR UPDATE OF m", (model_id,)
                    )
                ).fetchone()
                if row is None:
                    return None
                before = _record(row)
                if before.updated_at != expected_updated_at:
                    raise StaleEdit(before)
                merged = ModelRecord.model_validate({**before.model_dump(), **dict(changes)})
                if merged == before:
                    return before, before
                updated = await (
                    await conn.execute(
                        _UPDATE_MODEL,
                        (
                            merged.provider,
                            merged.model,
                            merged.voice,
                            Jsonb([voice.model_dump(mode="json") for voice in merged.voices]),
                            merged.creator,
                            merged.source,
                            merged.licensing,
                            merged.on_prem,
                            merged.region,
                            merged.arena_enabled,
                            merged.collected,
                            merged.published,
                            user_id,
                            email,
                            model_id,
                        ),
                    )
                ).fetchone()
                if updated is None:  # pragma: no cover — unreachable, the row is locked
                    raise RuntimeError("UPDATE benchmarks_v2.models returned no row")
                if merged.tags != before.tags:
                    await conn.execute(
                        "DELETE FROM benchmarks_v2.model_tags WHERE model_id = %s", (model_id,)
                    )
                    await self._write_tags(conn, model_id, merged.tags)
                after = merged.model_copy(
                    update={
                        "updated_by_user_id": user_id,
                        "updated_by_email": email,
                        "updated_at": updated["updated_at"],
                    }
                )
                await conn.execute(
                    _INSERT_HISTORY,
                    (
                        model_id,
                        after.modality,
                        after.provider,
                        after.model,
                        _snapshot(before),
                        _snapshot(after),
                        user_id,
                        org_id,
                        email,
                    ),
                )
        except psycopg.errors.UniqueViolation as exc:
            raise DuplicateKey(
                f"a model named ({before.modality}, {merged.provider}, {merged.model}) exists"
            ) from exc
        return before, after

    async def history(self, model_id: int, limit: int = 5) -> list[ModelChange]:
        """The newest *limit* history rows for one model, newest first."""
        sql = """
            SELECT * FROM benchmarks_v2.model_history
            WHERE model_id = %s
            ORDER BY changed_at DESC, id DESC
            LIMIT %s
        """
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, (model_id, limit))).fetchall()
        return [ModelChange.model_validate(dict(row)) for row in rows]

    async def recent_history(self, per_model: int = 5) -> dict[int, list[ModelChange]]:
        """The newest *per_model* history rows for every model that has any."""
        sql = """
            SELECT * FROM (
                SELECT h.*,
                       row_number() OVER (
                           PARTITION BY model_id ORDER BY changed_at DESC, id DESC
                       ) AS rn
                FROM benchmarks_v2.model_history h
            ) numbered
            WHERE rn <= %s
            ORDER BY model_id, changed_at DESC, id DESC
        """
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, (per_model,))).fetchall()
        history: dict[int, list[ModelChange]] = {}
        for row in rows:
            change = ModelChange.model_validate(dict(row))
            history.setdefault(change.model_id, []).append(change)
        return history

    @staticmethod
    async def _write_tags(
        conn: psycopg.AsyncConnection[psycopg.rows.DictRow],
        model_id: int,
        tags: tuple[str, ...],
    ) -> None:
        if not tags:
            return
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO benchmarks_v2.model_tags (model_id, tag) VALUES (%s, %s)",
                [(model_id, tag) for tag in tags],
            )
