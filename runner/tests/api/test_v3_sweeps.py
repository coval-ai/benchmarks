# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/sweeps over HTTP."""

from __future__ import annotations

from typing import Any

import psycopg
from httpx import AsyncClient

from tests.api.conftest import COVAL_ORG, EA_ORG, _make_db_url, bearer


def _half_done_sweep(postgresql: Any) -> int:
    """An unfinished sweep with two results, one item queued and one out of tries."""
    conn: Any
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        model_id = conn.execute(
            "SELECT id FROM benchmarks_v2.models WHERE provider = 'seed' AND model = 'stt'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO benchmarks_v2.benchmarks"
            " (slug, modality, name, visibility, updated_by_user_id)"
            " VALUES ('stt-wildasr', 'STT', 'WildASR', 'coval', 'tests')"
        )
        conn.execute(
            "INSERT INTO benchmarks_v2.benchmark_datasets (dataset_id, benchmark, condition)"
            " VALUES ('stt-wildasr-clean', 'stt-wildasr', 'clean')"
        )
        sweep_id = conn.execute(
            "INSERT INTO benchmarks_v2.sweeps"
            " (model_id, model_version, dataset_id, dataset_version)"
            " VALUES (%s, 1, 'stt-wildasr-clean', '1.1.0') RETURNING id",
            (model_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO benchmarks_v2.sweep_results (sweep_id, sample_id)"
            " VALUES (%(id)s, 'a'), (%(id)s, 'b')",
            {"id": sweep_id},
        )
        conn.execute(
            "INSERT INTO benchmarks_v2.sweep_queue (sweep_id, sample_id, tries_left)"
            " VALUES (%(id)s, 'c', 3), (%(id)s, 'd', 0)",
            {"id": sweep_id},
        )
    return int(model_id)


async def test_coval_sees_how_far_along_an_unfinished_sweep_is(
    client: AsyncClient, postgresql: Any
) -> None:
    model_id = _half_done_sweep(postgresql)

    response = await client.get("/v3/sweeps", headers=bearer(sub="u", org_id=COVAL_ORG))

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    (sweep,) = response.json()["sweeps"]
    assert sweep.pop("id")
    assert sweep.pop("created_at")
    assert sweep == {
        "model_id": model_id,
        "modality": "STT",
        "provider": "seed",
        "model": "stt",
        "model_version": 1,
        "benchmark": "stt-wildasr",
        "condition": "clean",
        "dataset_id": "stt-wildasr-clean",
        "dataset_version": "1.1.0",
        "completed_at": None,
        "result_count": 2,
        "queued_count": 1,
        "gave_up_count": 1,
    }


async def test_sweeps_are_for_coval_only(client: AsyncClient, postgresql: Any) -> None:
    _half_done_sweep(postgresql)

    anonymous = await client.get("/v3/sweeps")
    partner = await client.get("/v3/sweeps", headers=bearer(sub="u", org_id=EA_ORG))

    assert anonymous.status_code == 403
    assert partner.status_code == 403
    assert "sweeps" not in partner.json()
