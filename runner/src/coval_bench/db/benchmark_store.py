# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``BenchmarkStore``: reads and admin writes of the benchmark registry."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from coval_bench.registries import Visibility

BenchmarkPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]


class RegistryConflict(Exception):  # noqa: N818 — reads as a state, like HTTP 409
    """The write would break a registry rule; the message says which."""


class BenchmarkDataset(BaseModel):
    dataset_id: str
    condition: str | None


class RegisteredBenchmark(BaseModel):
    slug: str
    modality: str
    name: str
    visibility: Visibility
    datasets: list[BenchmarkDataset]
    updated_by_user_id: str
    updated_by_email: str | None
    updated_at: datetime


_SELECT = """
    SELECT b.slug, b.modality, b.name, b.visibility,
           b.updated_by_user_id, b.updated_by_email, b.updated_at,
           COALESCE(
             (SELECT jsonb_agg(jsonb_build_object('dataset_id', d.dataset_id,
                                                  'condition', d.condition)
                               ORDER BY d.condition NULLS FIRST, d.dataset_id)
              FROM benchmarks_v2.benchmark_datasets d WHERE d.benchmark = b.slug),
             '[]'::jsonb) AS datasets
    FROM benchmarks_v2.benchmarks b
"""
_TOUCH = """
    UPDATE benchmarks_v2.benchmarks
    SET updated_by_user_id = %s, updated_by_email = %s, updated_at = now()
    WHERE slug = %s
"""


class BenchmarkStore:
    """Per-pool persistence helper for the benchmark registry."""

    def __init__(self, pool: BenchmarkPool) -> None:
        self._pool = pool

    async def benchmarks(self) -> list[RegisteredBenchmark]:
        """Every benchmark with the datasets it owns."""
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(_SELECT + " ORDER BY b.modality, b.slug")).fetchall()
        return [RegisteredBenchmark.model_validate(dict(row)) for row in rows]

    async def _get(self, conn: psycopg.AsyncConnection[Any], slug: str) -> RegisteredBenchmark:
        row = await (await conn.execute(_SELECT + " WHERE b.slug = %s", (slug,))).fetchone()
        return RegisteredBenchmark.model_validate(dict(row or {}))

    async def create(
        self,
        *,
        slug: str,
        modality: str,
        name: str,
        visibility: Visibility,
        user_id: str,
        email: str | None,
    ) -> RegisteredBenchmark:
        try:
            async with self._pool.connection() as conn, conn.transaction():
                await conn.execute(
                    "INSERT INTO benchmarks_v2.benchmarks"
                    " (slug, modality, name, visibility, updated_by_user_id, updated_by_email)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    (slug, modality, name, visibility, user_id, email),
                )
                return await self._get(conn, slug)
        except psycopg.errors.UniqueViolation as exc:
            raise RegistryConflict(f"benchmark {slug!r} already exists") from exc

    async def set_visibility(
        self, slug: str, visibility: Visibility, *, user_id: str, email: str | None
    ) -> RegisteredBenchmark | None:
        """Change who sees the benchmark's results; None when there is no such benchmark."""
        async with self._pool.connection() as conn, conn.transaction():
            updated = await conn.execute(
                "UPDATE benchmarks_v2.benchmarks SET visibility = %s, updated_by_user_id = %s,"
                " updated_by_email = %s, updated_at = now() WHERE slug = %s",
                (visibility, user_id, email, slug),
            )
            return await self._get(conn, slug) if updated.rowcount else None

    async def put_dataset(
        self, slug: str, dataset_id: str, *, condition: str | None, user_id: str, email: str | None
    ) -> RegisteredBenchmark | None:
        """Give *slug* the dataset under *condition*, or relabel one it already owns.

        None when there is no such benchmark. A dataset another benchmark owns, or a
        condition another of this benchmark's datasets holds, is a conflict.
        """
        try:
            async with self._pool.connection() as conn, conn.transaction():
                if not (await conn.execute(_TOUCH, (user_id, email, slug))).rowcount:
                    return None
                written = await conn.execute(
                    """
                    INSERT INTO benchmarks_v2.benchmark_datasets (dataset_id, benchmark, condition)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (dataset_id) DO UPDATE SET condition = EXCLUDED.condition
                    WHERE benchmark_datasets.benchmark = EXCLUDED.benchmark
                    """,
                    (dataset_id, slug, condition),
                )
                if not written.rowcount:
                    raise RegistryConflict(f"dataset {dataset_id!r} belongs to another benchmark")
                return await self._get(conn, slug)
        except psycopg.errors.UniqueViolation as exc:
            held = "its one unconditioned dataset" if condition is None else repr(condition)
            raise RegistryConflict(f"benchmark {slug!r} already has {held}") from exc

    async def remove_dataset(
        self, slug: str, dataset_id: str, *, user_id: str, email: str | None
    ) -> RegisteredBenchmark | None:
        """Detach a dataset; None when *slug* doesn't own it. Results are untouched."""
        async with self._pool.connection() as conn, conn.transaction():
            deleted = await conn.execute(
                "DELETE FROM benchmarks_v2.benchmark_datasets"
                " WHERE dataset_id = %s AND benchmark = %s",
                (dataset_id, slug),
            )
            if not deleted.rowcount:
                return None
            await conn.execute(_TOUCH, (user_id, email, slug))
            return await self._get(conn, slug)
