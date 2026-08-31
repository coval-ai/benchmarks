# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``fetch_models`` reproduces the code registry from the seeded tables.

The comparison is the gate on switching every consumer over: if the database
answers differ from the literals in any field, the switch would change
behaviour. Both sides are sorted by natural key — post-seed models take ids at
the end, so id order diverges from literal order. Deleted with the registry it
compares against.
"""

from __future__ import annotations

import asyncio
from typing import Any

import psycopg
from pytest_postgresql.factories import postgresql

from coval_bench.db.registry_store import fetch_models
from coval_bench.registries import MODEL_REGISTRY, ModelStatus, RegisteredModel

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
    """The model normalized to what consumers actually distinguish.

    Two differences are expected and behaviour-neutral. ``model_tags`` has no
    position column, so the database returns a model's tags alphabetized while
    the literals carry the order they were typed in — nothing reads that order,
    the site tests tag membership per facet. And the two state booleans cannot
    tell ``PENDING`` from ``RETIRED``, which every consumer already treats
    alike: both are excluded from collection and marked disabled.
    """
    status = ModelStatus.RETIRED if model.status is ModelStatus.PENDING else model.status
    return model.model_copy(update={"tags": tuple(sorted(model.tags)), "status": status})


def _key(model: RegisteredModel) -> tuple[str, str, str]:
    return (model.benchmark.value, model.provider, model.model)


def test_the_database_reproduces_the_registry(fetch_pg: psycopg.Connection[Any]) -> None:
    models = _fetched(fetch_pg)
    assert len(models) == len(MODEL_REGISTRY)
    literals = sorted(MODEL_REGISTRY, key=_key)
    for fetched, literal in zip(sorted(models, key=_key), literals, strict=True):
        assert _comparable(fetched) == _comparable(literal), f"{literal.provider}/{literal.model}"
