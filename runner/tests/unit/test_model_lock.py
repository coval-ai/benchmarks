# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL checks for the cross-job model lock."""

from __future__ import annotations

from typing import Any

from pytest_postgresql.factories import postgresql

from coval_bench.db.model_lock import model_claim, model_lock, model_lock_key
from coval_bench.registries import Benchmark, RegisteredModel
from tests.unit.conftest import open_pool

pg_conn = postgresql("pg_proc")


async def test_a_held_model_times_out_for_the_next_job_until_released(pg_conn: Any) -> None:
    pool = await open_pool(pg_conn)
    try:
        async with model_lock(pool, "model_lock:acme/fast-1", timeout_s=0.05) as first:
            async with model_lock(pool, "model_lock:acme/fast-1", timeout_s=0.05) as second:
                assert (first, second) == (True, False)
            async with model_lock(pool, "model_lock:acme/fast-2", timeout_s=0.05) as other:
                assert other is True
        async with model_lock(pool, "model_lock:acme/fast-1", timeout_s=0.05) as again:
            assert again is True
    finally:
        await pool.close()


async def test_a_claimed_model_is_refused_at_once_but_can_still_be_locked(pg_conn: Any) -> None:
    pool = await open_pool(pg_conn)
    key = "model_lock:acme/fast-1"
    try:
        async with model_claim(pool, key) as first:
            async with model_claim(pool, key) as second:
                assert (first, second) == (True, False)
            async with model_lock(pool, key, timeout_s=0.05) as locked:
                assert locked is True
        async with model_claim(pool, key) as again:
            assert again is True
    finally:
        await pool.close()


def test_a_single_connection_provider_shares_one_lock_across_its_models() -> None:
    class OneAtATime:
        _SINGLE_CONNECTION = True

    def key(model: str, provider_cls: Any) -> str:
        entry = RegisteredModel(
            benchmark=Benchmark.STT, provider="acme", model=model, collected=True, published=True
        )
        return model_lock_key(entry, provider_cls)

    assert key("a", OneAtATime) == key("b", OneAtATime)
    assert key("a", object) != key("b", object)
