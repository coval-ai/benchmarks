# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/quality over HTTP."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from httpx import AsyncClient

from coval_bench.registries import RegisteredModel
from coval_bench.registries.benchmarks import Benchmark
from tests.api.conftest import (
    COVAL_ORG,
    EA_MODEL,
    EA_PROVIDER,
    _make_db_url,
    add_models,
    bearer,
    set_company_org,
)


@pytest.fixture(autouse=True)
def registry(postgresql: Any) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute(
            """INSERT INTO benchmarks_v2.benchmarks
                   (slug, modality, name, visibility, updated_by_user_id)
               VALUES ('stt-v3', 'STT', 'STT v3', 'public', 'tests'),
                      ('stt-wildasr', 'STT', 'WildASR', 'public', 'tests'),
                      ('tts-v2', 'TTS', 'TTS v2', 'public', 'tests')"""
        )
        conn.execute(
            """INSERT INTO benchmarks_v2.benchmark_datasets (dataset_id, benchmark, condition)
               VALUES ('stt-v3', 'stt-v3', NULL),
                      ('stt-wildasr-clean', 'stt-wildasr', 'clean'),
                      ('stt-wildasr-farfield', 'stt-wildasr', 'farfield'),
                      ('tts-v2', 'tts-v2', NULL)"""
        )


def _model_id(conn: Any, provider: str, model: str) -> int:
    row = conn.execute(
        "SELECT id FROM benchmarks_v2.models WHERE provider = %s AND model = %s", (provider, model)
    ).fetchone()
    return int(row[0])


def _sweep(
    conn: Any,
    model_id: int,
    scores: list[tuple[str, str, int, int]],
    *,
    dataset: str = "stt-v3",
    model_version: int = 1,
    completed: bool = True,
    gave_up: int = 0,
) -> None:
    """A sweep with one scored result per (sample, voice, errors, reference units)."""
    sweep_id = conn.execute(
        "INSERT INTO benchmarks_v2.sweeps"
        " (model_id, model_version, dataset_id, dataset_version, completed_at)"
        " VALUES (%s, %s, %s, '1.1.0', CASE WHEN %s THEN now() END) RETURNING id",
        (model_id, model_version, dataset, completed),
    ).fetchone()[0]
    for sample, voice, errors, units in scores:
        result_id = conn.execute(
            "INSERT INTO benchmarks_v2.sweep_results (sweep_id, sample_id, voice)"
            " VALUES (%s, %s, %s) RETURNING id",
            (sweep_id, sample, voice),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO benchmarks_v2.sweep_scores"
            " (sweep_result_id, metric, metric_version, errors, reference_units)"
            " VALUES (%s, 'WER', '4', %s, %s)",
            (result_id, errors, units),
        )
    for index in range(gave_up):
        conn.execute(
            "INSERT INTO benchmarks_v2.sweep_queue (sweep_id, sample_id, tries_left)"
            " VALUES (%s, %s, 0)",
            (sweep_id, f"gave-up-{index}"),
        )


async def test_quality_is_pooled_over_the_latest_completed_sweep(
    client: AsyncClient, postgresql: Any
) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        stt = _model_id(conn, "seed", "stt")
        tts = _model_id(conn, "seed", "tts-a")
        _sweep(conn, stt, [("a", "", 1, 10), ("b", "", 2, 30)], gave_up=1)
        _sweep(conn, stt, [("a", "", 9, 10)], dataset="stt-wildasr-clean", completed=False)
        _sweep(conn, tts, [("a", "f", 0, 20), ("a", "m", 1, 20)], dataset="tts-v2")

    response = await client.get("/v3/quality")

    assert response.status_code == 200
    rows = {(row["model_id"], row["benchmark"]): row for row in response.json()["quality"]}
    assert set(rows) == {(stt, "stt-v3"), (tts, "tts-v2")}
    measured_at = rows[(stt, "stt-v3")].pop("measured_at")
    assert measured_at.endswith("Z")
    assert rows[(stt, "stt-v3")] == {
        "model_id": stt,
        "provider": "seed",
        "model": "stt",
        "model_version": 1,
        "benchmark": "stt-v3",
        "benchmark_version": "1.1.0",
        "condition": None,
        "metric": "WER",
        "metric_version": "4",
        "value": 7.5,
        "unit": "percent",
        "errors": 3,
        "reference_units": 40,
        "sample_count": 2,
        "expected_sample_count": 3,
    }
    pooled = rows[(tts, "tts-v2")]
    assert (pooled["value"], pooled["errors"], pooled["sample_count"]) == (2.5, 1, 2)
    only_tts = (await client.get("/v3/quality?modality=TTS")).json()["quality"]
    assert [row["model_id"] for row in only_tts] == [tts]


async def test_conditions_pool_into_one_row_unless_broken_down(
    client: AsyncClient, postgresql: Any
) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        stt = _model_id(conn, "seed", "stt")
        _sweep(conn, stt, [("a", "", 1, 10)])
        _sweep(conn, stt, [("a", "", 2, 10)], dataset="stt-wildasr-clean")
        _sweep(conn, stt, [("a", "", 5, 30), ("b", "", 0, 10)], dataset="stt-wildasr-farfield")
        _sweep(conn, stt, [("a", "", 9, 10)], dataset="stt-unregistered")

    pooled = (await client.get("/v3/quality")).json()["quality"]
    wildasr = (await client.get("/v3/quality?benchmark=stt-wildasr")).json()["quality"]
    by_condition = (await client.get("/v3/quality?breakdown=condition")).json()["quality"]

    assert [(row["benchmark"], row["condition"], row["value"]) for row in pooled] == [
        ("stt-v3", None, 10.0),
        ("stt-wildasr", None, 14.0),
    ]
    assert [(row["errors"], row["reference_units"], row["sample_count"]) for row in wildasr] == [
        (7, 50, 3)
    ]
    assert [(row["benchmark"], row["condition"], row["value"]) for row in by_condition] == [
        ("stt-v3", None, 10.0),
        ("stt-wildasr", "clean", 20.0),
        ("stt-wildasr", "farfield", 12.5),
    ]


async def test_quality_filters_by_model_and_metric(client: AsyncClient, postgresql: Any) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        stt = _model_id(conn, "seed", "stt")
        tts = _model_id(conn, "seed", "tts-a")
        _sweep(conn, stt, [("a", "", 1, 10)])
        _sweep(conn, tts, [("a", "f", 0, 20)], dataset="tts-v2")

    one_model = (await client.get(f"/v3/quality?model_id={tts}")).json()["quality"]
    wer = (await client.get("/v3/quality?metric=WER")).json()["quality"]
    other_metric = (await client.get("/v3/quality?metric=TTFS")).json()["quality"]
    bad_id = await client.get("/v3/quality?model_id=abc")

    assert [row["model_id"] for row in one_model] == [tts]
    assert {row["model_id"] for row in wer} == {stt, tts}
    assert other_metric == []
    assert bad_id.status_code == 400


async def test_a_benchmark_stays_on_the_version_that_finished_every_condition(
    client: AsyncClient, postgresql: Any
) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        stt = _model_id(conn, "seed", "stt")
        _sweep(conn, stt, [("a", "", 2, 10)], dataset="stt-wildasr-clean")
        unfinished = (await client.get("/v3/quality?breakdown=condition")).json()["quality"]
        _sweep(conn, stt, [("a", "", 4, 10)], dataset="stt-wildasr-farfield")
        _sweep(conn, stt, [("a", "", 0, 10)], dataset="stt-wildasr-clean", model_version=2)

    rows = (await client.get("/v3/quality")).json()["quality"]

    assert unfinished == []
    assert [(row["model_version"], row["value"]) for row in rows] == [(1, 30.0)]


async def test_a_model_keeps_its_previous_score_while_a_new_version_is_swept(
    client: AsyncClient, postgresql: Any
) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        stt = _model_id(conn, "seed", "stt")
        _sweep(conn, stt, [("a", "", 1, 10)])
        _sweep(conn, stt, [("a", "", 5, 10)], model_version=2, completed=False)

    before = (await client.get("/v3/quality")).json()["quality"]
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute("UPDATE benchmarks_v2.sweeps SET completed_at = now() WHERE model_version = 2")
    after = (await client.get("/v3/quality")).json()["quality"]

    assert [(row["model_version"], row["value"]) for row in before] == [(1, 10.0)]
    assert [(row["model_version"], row["value"]) for row in after] == [(2, 50.0)]


async def test_an_unpublished_model_is_only_in_its_own_orgs_quality(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(
        postgresql,
        RegisteredModel(
            benchmark=Benchmark.STT,
            provider=EA_PROVIDER,
            model=EA_MODEL,
            collected=True,
            visibility="partner",
        ),
    )
    set_company_org(postgresql, EA_PROVIDER, "org_test_linked")
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        _sweep(conn, _model_id(conn, EA_PROVIDER, EA_MODEL), [("a", "", 1, 10)])

    public = await client.get("/v3/quality")
    partner = await client.get("/v3/quality", headers=bearer(sub="u", org_id="org_test_linked"))

    assert public.json()["quality"] == []
    assert [row["provider"] for row in partner.json()["quality"]] == [EA_PROVIDER]


async def test_a_partner_level_benchmark_shows_only_on_the_callers_own_models(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(
        postgresql,
        RegisteredModel(
            benchmark=Benchmark.STT,
            provider=EA_PROVIDER,
            model=EA_MODEL,
            collected=True,
            visibility="public",
        ),
    )
    set_company_org(postgresql, EA_PROVIDER, "org_test_linked")
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        _sweep(conn, _model_id(conn, EA_PROVIDER, EA_MODEL), [("a", "", 1, 10)])
        _sweep(conn, _model_id(conn, "seed", "stt"), [("a", "", 1, 10)])
        conn.execute("UPDATE benchmarks_v2.benchmarks SET visibility = 'partner'")

    async def providers(headers: dict[str, str]) -> set[str]:
        rows = (await client.get("/v3/quality", headers=headers)).json()["quality"]
        return {row["provider"] for row in rows}

    assert await providers({}) == set()
    assert await providers(bearer(sub="u", org_id="org_test_linked")) == {EA_PROVIDER}
    assert await providers(bearer(org_id=COVAL_ORG)) == {EA_PROVIDER, "seed"}


async def test_a_completed_sweep_with_nothing_scored_keeps_the_previous_score(
    client: AsyncClient, postgresql: Any
) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        stt = _model_id(conn, "seed", "stt")
        _sweep(conn, stt, [("a", "", 1, 10)])
        _sweep(conn, stt, [], model_version=2, gave_up=3)

    rows = (await client.get("/v3/quality")).json()["quality"]

    assert [(row["model_version"], row["value"]) for row in rows] == [(1, 10.0)]


async def test_a_coval_only_metric_is_left_out_for_the_public(
    client: AsyncClient, postgresql: Any
) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        _sweep(conn, _model_id(conn, "seed", "stt"), [("a", "", 1, 10)])
        conn.execute("UPDATE benchmarks_v2.metrics SET visibility = 'coval' WHERE code = 'WER'")

    public = await client.get("/v3/quality")
    internal = await client.get("/v3/quality", headers=bearer(org_id=COVAL_ORG))

    assert public.json()["quality"] == []
    assert [row["metric"] for row in internal.json()["quality"]] == ["WER"]


async def test_a_metric_with_no_registry_row_is_served_to_no_one(
    client: AsyncClient, postgresql: Any
) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        _sweep(conn, _model_id(conn, "seed", "stt"), [("a", "", 1, 10)])
        conn.execute("UPDATE benchmarks_v2.sweep_scores SET metric = 'Unregistered'")

    response = await client.get("/v3/quality", headers=bearer(org_id=COVAL_ORG))

    assert response.json()["quality"] == []
