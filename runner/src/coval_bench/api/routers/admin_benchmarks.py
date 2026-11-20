# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Admin benchmark registry endpoints: which benchmark and condition a dataset id belongs to.

Coval-org only (``require_coval_admin``). These rows are labels: they don't build a
dataset or schedule a run, and detaching a dataset leaves its results alone. There is
no DELETE for a benchmark.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Response
from psycopg_pool import AsyncConnectionPool

from coval_bench.api.clerk import CovalAdmin
from coval_bench.api.deps import get_pool, require_coval_admin
from coval_bench.api.internal import never_shared
from coval_bench.api.schemas import (
    AdminBenchmarkCreate,
    AdminBenchmarkDatasetPut,
    AdminBenchmarkPatch,
    AdminBenchmarksResponse,
)
from coval_bench.db.benchmark_store import BenchmarkStore, RegisteredBenchmark, RegistryConflict

router = APIRouter(tags=["admin"], dependencies=[Depends(require_coval_admin)])


async def _guarded[T](call: Awaitable[T]) -> T:
    """Run a store call; a missing table or grant is a 503, a broken rule a 409."""
    try:
        return await call
    except RegistryConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(503, "the benchmark registry is unavailable") from exc


def _found(benchmark: RegisteredBenchmark | None, what: str) -> RegisteredBenchmark:
    if benchmark is None:
        raise HTTPException(404, f"no such {what}")
    return benchmark


@router.get("/admin/benchmarks", response_model=AdminBenchmarksResponse)
async def list_admin_benchmarks(
    response: Response, pool: AsyncConnectionPool[Any] = Depends(get_pool)
) -> AdminBenchmarksResponse:
    """Every benchmark with the datasets it owns."""
    never_shared(response)
    return AdminBenchmarksResponse(benchmarks=await _guarded(BenchmarkStore(pool).benchmarks()))


@router.post("/admin/benchmarks", response_model=RegisteredBenchmark, status_code=201)
async def create_admin_benchmark(
    body: AdminBenchmarkCreate,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> RegisteredBenchmark:
    """Register a benchmark. It owns no datasets until they are added."""
    never_shared(response)
    return await _guarded(
        BenchmarkStore(pool).create(
            slug=body.slug,
            modality=str(body.modality),
            name=body.name,
            visibility=body.visibility,
            user_id=admin.user_id,
            email=admin.email,
        )
    )


@router.patch("/admin/benchmarks/{slug}", response_model=RegisteredBenchmark)
async def update_admin_benchmark(
    slug: str,
    body: AdminBenchmarkPatch,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> RegisteredBenchmark:
    """Change who sees the benchmark's results."""
    never_shared(response)
    return _found(
        await _guarded(
            BenchmarkStore(pool).set_visibility(
                slug, body.visibility, user_id=admin.user_id, email=admin.email
            )
        ),
        "benchmark",
    )


@router.put("/admin/benchmarks/{slug}/datasets/{dataset_id}", response_model=RegisteredBenchmark)
async def put_admin_benchmark_dataset(
    slug: str,
    dataset_id: str,
    body: AdminBenchmarkDatasetPut,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> RegisteredBenchmark:
    """Add a dataset to the benchmark, or change the condition of one it owns."""
    never_shared(response)
    return _found(
        await _guarded(
            BenchmarkStore(pool).put_dataset(
                slug,
                dataset_id,
                condition=body.condition,
                user_id=admin.user_id,
                email=admin.email,
            )
        ),
        "benchmark",
    )


@router.delete("/admin/benchmarks/{slug}/datasets/{dataset_id}", response_model=RegisteredBenchmark)
async def delete_admin_benchmark_dataset(
    slug: str,
    dataset_id: str,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> RegisteredBenchmark:
    """Detach a dataset from the benchmark."""
    never_shared(response)
    return _found(
        await _guarded(
            BenchmarkStore(pool).remove_dataset(
                slug, dataset_id, user_id=admin.user_id, email=admin.email
            )
        ),
        "dataset on that benchmark",
    )
