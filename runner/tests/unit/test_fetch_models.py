# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``fetch_models`` returns a superset of the code registry from the seeded tables.

Every literal must come back from the database equal in every field, or the
switch to database-backed consumers would change behaviour. The database may
also hold rows the code registry never had (models seeded only by migration
now that the database is authoritative); those are pinned so a drifting
literal still fails here. Deleted with the registry it compares against.
"""

from __future__ import annotations

import asyncio
from typing import Any

import psycopg
from pytest_postgresql.factories import postgresql

from coval_bench.db.registry_store import fetch_models
from coval_bench.registries import MODEL_REGISTRY, RegisteredModel

from .conftest import apply_migrations, open_pool

fetch_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB


def _fetched(conn: psycopg.Connection[Any]) -> list[Any]:
    apply_migrations(conn)

    async def _run() -> list[Any]:
        pool = await open_pool(conn)
        try:
            return await fetch_models(pool)
        finally:
            await pool.close()

    return asyncio.run(_run())


def _comparable(model: RegisteredModel) -> RegisteredModel:
    """The model with tags in a canonical order.

    ``model_tags`` has no position column, so the database returns a model's
    tags alphabetized while the literals carry the order they were typed in.
    Nothing reads that order: the site tests tag membership per facet.
    """
    return model.model_copy(update={"tags": tuple(sorted(model.tags))})


def _key(model: RegisteredModel) -> tuple[str, str, str]:
    return (model.benchmark.value, model.provider, model.model)


MIGRATION_ONLY_MODELS = {("LLM", "phonely", "phonely-agent")}


def test_the_database_is_a_superset_of_the_registry(fetch_pg: psycopg.Connection[Any]) -> None:
    fetched = {_key(m): _comparable(m) for m in _fetched(fetch_pg)}
    for literal in MODEL_REGISTRY:
        assert fetched[_key(literal)] == _comparable(literal), f"{literal.provider}/{literal.model}"
    assert fetched.keys() - {_key(m) for m in MODEL_REGISTRY} == MIGRATION_ONLY_MODELS
