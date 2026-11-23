# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/{modality}/sweeps over HTTP."""

from __future__ import annotations

from typing import Any

import psycopg
from httpx import AsyncClient

from coval_bench.registries import RegisteredModel
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import Architecture
from tests.api.conftest import COVAL_ORG, EA_ORG, _make_db_url, add_models, bearer


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
        result_id = conn.execute(
            "INSERT INTO benchmarks_v2.sweep_results"
            " (sweep_id, sample_id, transcript, timings, audio_uri)"
            " VALUES (%s, 'a', 'hello there', '{\"ttft_ms\": 120}', 'gs://bucket/a.wav')"
            " RETURNING id",
            (sweep_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO benchmarks_v2.sweep_scores"
            " (sweep_result_id, metric, metric_version, errors, reference_units)"
            " VALUES (%s, 'WER', '4', 1, 2)",
            (result_id,),
        )
        conn.execute(
            "INSERT INTO benchmarks_v2.sweep_results (sweep_id, sample_id) VALUES (%s, 'b')",
            (sweep_id,),
        )
        conn.execute(
            "INSERT INTO benchmarks_v2.sweep_queue (sweep_id, sample_id, tries_left, last_error)"
            " VALUES (%(id)s, 'c', 3, NULL), (%(id)s, 'd', 0, 'timed out')",
            {"id": sweep_id},
        )
    return int(model_id)


async def test_coval_sees_how_far_along_an_unfinished_sweep_is(
    client: AsyncClient, postgresql: Any
) -> None:
    model_id = _half_done_sweep(postgresql)

    response = await client.get("/v3/stt/sweeps", headers=bearer(sub="u", org_id=COVAL_ORG))

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
        "calling_count": 0,
        "scoring_count": 0,
        "gave_up_count": 1,
    }


async def test_coval_sees_each_item_of_a_sweep_whether_or_not_it_ran(
    client: AsyncClient, postgresql: Any
) -> None:
    _half_done_sweep(postgresql)
    staff = bearer(sub="u", org_id=COVAL_ORG)
    (sweep,) = (await client.get("/v3/stt/sweeps", headers=staff)).json()["sweeps"]
    url = f"/v3/stt/sweeps/{sweep['id']}/items"

    items = (await client.get(url, headers=staff)).json()["items"]
    missing = await client.get("/v3/stt/sweeps/999999/items", headers=staff)
    other_modality = await client.get(f"/v3/tts/sweeps/{sweep['id']}/items", headers=staff)
    assert (await client.get("/v3/tts/sweeps", headers=staff)).json()["sweeps"] == []
    assert other_modality.status_code == 404

    assert [(item["sample_id"], item["status"]) for item in items] == [
        ("a", "done"),
        ("b", "done"),
        ("c", "queued"),
        ("d", "gave_up"),
    ]
    assert items[0].pop("ran_at")
    assert items[0] == {
        "sample_id": "a",
        "voice": None,
        "status": "done",
        "transcript": "hello there",
        "timings": {"ttft_ms": 120},
        "audio_uri": "gs://bucket/a.wav",
        "tries_left": None,
        "last_error": None,
        "scores": [{"metric": "WER", "metric_version": "4", "errors": 1, "reference_units": 2}],
    }
    assert items[3] == {
        "sample_id": "d",
        "voice": None,
        "status": "gave_up",
        "transcript": None,
        "timings": None,
        "audio_uri": None,
        "ran_at": None,
        "tries_left": 0,
        "last_error": "timed out",
        "scores": [],
    }
    assert missing.status_code == 404


async def test_sweeps_are_for_coval_only(client: AsyncClient, postgresql: Any) -> None:
    _half_done_sweep(postgresql)

    for url in ("/v3/stt/sweeps", "/v3/stt/sweeps/1/items", "/v3/s2s/sweeps/1/calls"):
        anonymous = await client.get(url)
        partner = await client.get(url, headers=bearer(sub="u", org_id=EA_ORG))

        assert anonymous.status_code == 403
        assert partner.status_code == 403
        assert set(partner.json()) == {"error"}


def _s2s_sweep(postgresql: Any) -> int:
    """An S2S sweep with one call in each place a call can be."""
    add_models(
        postgresql,
        RegisteredModel(
            benchmark=Benchmark.S2S,
            provider="acme",
            model="live-1",
            collected=True,
            visibility="coval",
            architecture=Architecture.SPEECH_TO_SPEECH,
        ),
    )
    conn: Any
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        sweep_id = conn.execute(
            "INSERT INTO benchmarks_v2.sweeps"
            " (model_id, model_version, dataset_id, dataset_version)"
            " SELECT id, 1, 'id_roulette', 'abc' FROM benchmarks_v2.models"
            " WHERE provider = 'acme' AND model = 'live-1' RETURNING id"
        ).fetchone()[0]
        result_id = conn.execute(
            "INSERT INTO benchmarks_v2.s2s_results"
            " (sweep_id, test_case, persona, repeat, simulation_id)"
            " VALUES (%s, 'card-1', 'quiet room', 1, 'sim-done') RETURNING id",
            (sweep_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO benchmarks_v2.s2s_scores"
            " (s2s_result_id, metric, metric_version, value, reference_units)"
            " VALUES (%s, 'CallLength', '1', 41.5, 1)",
            (result_id,),
        )
        conn.execute(
            "INSERT INTO benchmarks_v2.s2s_queue"
            " (sweep_id, test_case, persona, repeat, tries_left, last_error,"
            "  claimed_at, simulation_id)"
            " VALUES (%(id)s, 'card-2', 'quiet room', 1, 3, NULL, NULL, NULL),"
            "        (%(id)s, 'card-3', 'quiet room', 1, 3, NULL, now(), NULL),"
            "        (%(id)s, 'card-4', 'quiet room', 1, 2, 'hung up', now(), 'sim-made'),"
            "        (%(id)s, 'card-5', 'quiet room', 1, 0, 'no answer', NULL, NULL)",
            {"id": sweep_id},
        )
    return int(sweep_id)


async def test_coval_sees_where_each_call_of_an_s2s_sweep_is(
    client: AsyncClient, postgresql: Any
) -> None:
    _half_done_sweep(postgresql)
    sweep_id = _s2s_sweep(postgresql)
    staff = bearer(sub="u", org_id=COVAL_ORG)

    (sweep,) = (await client.get("/v3/s2s/sweeps", headers=staff)).json()["sweeps"]
    response = await client.get(f"/v3/s2s/sweeps/{sweep_id}/calls", headers=staff)

    assert (sweep["id"], sweep["dataset_id"], sweep["benchmark"]) == (sweep_id, "id_roulette", None)
    counts = ("result_count", "queued_count", "calling_count", "scoring_count", "gave_up_count")
    assert [sweep[count] for count in counts] == [1, 1, 1, 1, 1]
    assert response.status_code == 200
    calls = response.json()["calls"]
    assert [(call["test_case"], call["status"]) for call in calls] == [
        ("card-1", "done"),
        ("card-2", "queued"),
        ("card-3", "calling"),
        ("card-4", "scoring"),
        ("card-5", "gave_up"),
    ]
    done, _, calling, scoring, gave_up = calls
    assert done.pop("ran_at")
    assert done == {
        "test_case": "card-1",
        "persona": "quiet room",
        "repeat": 1,
        "status": "done",
        "simulation_id": "sim-done",
        "claimed_at": None,
        "tries_left": None,
        "last_error": None,
        "scores": [
            {"metric": "CallLength", "metric_version": "1", "value": 41.5, "reference_units": 1}
        ],
    }
    assert calling["claimed_at"] and calling["simulation_id"] is None
    assert (scoring["simulation_id"], scoring["last_error"]) == ("sim-made", "hung up")
    assert (gave_up["tries_left"], gave_up["last_error"]) == (0, "no answer")


async def test_only_an_s2s_sweep_has_calls(client: AsyncClient, postgresql: Any) -> None:
    _half_done_sweep(postgresql)
    sweep_id = _s2s_sweep(postgresql)
    staff = bearer(sub="u", org_id=COVAL_ORG)

    (stt,) = (await client.get("/v3/stt/sweeps", headers=staff)).json()["sweeps"]
    wrong_modality = await client.get(f"/v3/stt/sweeps/{sweep_id}/calls", headers=staff)
    not_s2s = await client.get(f"/v3/s2s/sweeps/{stt['id']}/calls", headers=staff)

    assert wrong_modality.status_code == 404
    assert not_s2s.status_code == 404
