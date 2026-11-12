# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Opening sweeps, against PostgreSQL."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.config import Settings
from coval_bench.db.registry_store import NewModel, RegistryStore
from coval_bench.registries import Benchmark
from coval_bench.runner import sweep
from tests.unit.conftest import apply_migrations, open_pool

pg_conn = postgresql("pg_proc")

_SETTINGS = cast(Settings, SimpleNamespace())
_ONLY = frozenset({"acme/fast-1"})


async def _pool_with_model(conn: Any) -> Any:
    apply_migrations(conn)
    pool = await open_pool(conn)
    new = NewModel(modality=Benchmark.STT, provider="acme", model="fast-1")
    await RegistryStore(pool).insert_model(new, user_id="user_editor", org_id=None, email=None)
    return pool


def _wire(monkeypatch: pytest.MonkeyPatch, pool: Any, versions: dict[str, str]) -> None:
    @asynccontextmanager
    async def fake_pool(_settings: Any) -> Any:
        yield pool

    monkeypatch.setattr(sweep, "lifespan_pool", fake_pool)
    monkeypatch.setattr(sweep, "SWEEP_STT_DATASETS", tuple(versions))
    monkeypatch.setattr(
        sweep, "dataset_manifest", lambda dataset_id: SimpleNamespace(version=versions[dataset_id])
    )


def _sweeps(conn: Any) -> list[Any]:
    conn.autocommit = True
    return list(
        conn.execute(
            "SELECT model_version, dataset_id, dataset_version"
            " FROM benchmarks_v2.sweeps ORDER BY id"
        ).fetchall()
    )


async def test_a_run_opens_one_sweep_per_dataset_and_a_rerun_opens_none(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        _wire(monkeypatch, pool, {"ds-1": "1.0.0", "ds-2": "1.0.0"})
        first = await sweep.open_sweeps(settings=_SETTINGS, only=_ONLY)
        second = await sweep.open_sweeps(settings=_SETTINGS, only=_ONLY)
    finally:
        await pool.close()

    assert (first, second) == ((1, 2), (1, 0))
    assert _sweeps(pg_conn) == [(1, "ds-1", "1.0.0"), (1, "ds-2", "1.0.0")]


async def test_a_model_version_bump_or_a_new_dataset_version_opens_a_new_sweep(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        versions = {"ds-1": "1.0.0"}
        _wire(monkeypatch, pool, versions)
        await sweep.open_sweeps(settings=_SETTINGS, only=_ONLY)
        versions["ds-1"] = "1.1.0"
        redone = await sweep.open_sweeps(settings=_SETTINGS, only=_ONLY)
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE benchmarks_v2.models SET version = 2 WHERE provider = 'acme'"
            )
        bumped = await sweep.open_sweeps(settings=_SETTINGS, only=_ONLY)
    finally:
        await pool.close()

    assert (redone, bumped) == ((1, 1), (1, 1))
    assert _sweeps(pg_conn) == [(1, "ds-1", "1.0.0"), (1, "ds-1", "1.1.0"), (2, "ds-1", "1.1.0")]
