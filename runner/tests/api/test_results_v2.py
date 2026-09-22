# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import psycopg
import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import create_engine

from coval_bench.api.app import create_app
from coval_bench.api.results_cursor import decode as decode_cursor
from coval_bench.api.results_cursor import encode as encode_cursor
from coval_bench.config import Settings
from coval_bench.registries import Benchmark, RegisteredModel
from tests.api.conftest import (
    COVAL_ORG,
    EA_MODEL,
    EA_ORG,
    EA_PROVIDER,
    _insert_result,
    _insert_run,
    _make_db_url,
    add_models,
    bearer,
)


def _seed_completed_evaluation(
    conn: psycopg.Connection[Any],
    run_id: int,
    *,
    captured_at: datetime,
    sample_id: str,
) -> UUID:
    """Insert one completed metric evaluation using the current schema."""
    observation_id = uuid4()
    evaluation_id = uuid4()
    conn.execute(
        """
        INSERT INTO benchmarks_v2.benchmark_observations
          (id, run_id, dataset_id, sample_id, provider, model, voice, benchmark,
           captured_at, status)
        VALUES (%s, %s, 'historical-dataset', %s, 'historical-provider',
                'historical-model', 'voice-a', 'STT', %s, 'succeeded')
        """,
        (observation_id, run_id, sample_id, captured_at),
    )
    conn.execute(
        """
        INSERT INTO benchmarks_v2.metric_evaluations
          (id, observation_id, metric_type, metric_version, evaluation_variant, status)
        VALUES (%s, %s, 'WER', 'v1', 'default', 'queued')
        """,
        (evaluation_id, observation_id),
    )
    conn.execute(
        "UPDATE benchmarks_v2.metric_evaluations SET status='running' WHERE id=%s",
        (evaluation_id,),
    )
    conn.execute(
        """
        INSERT INTO benchmarks_v2.metric_values
          (metric_evaluation_id, value_key, unit, value, value_role)
        VALUES (%s, 'primary', 'percent', 3, 'primary')
        """,
        (evaluation_id,),
    )
    conn.execute(
        "UPDATE benchmarks_v2.metric_evaluations SET status='succeeded' WHERE id=%s",
        (evaluation_id,),
    )
    return evaluation_id


def _apply_metric_id_enforcement(dsn: str) -> None:
    migration = import_module(
        "coval_bench.db.migrations.versions.20260917_0039_normalized_metric_id_not_null"
    )
    engine = create_engine(dsn.replace("postgresql://", "postgresql+psycopg://"))
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            operations = Operations(context)
            with patch.object(migration, "op", operations):
                migration.upgrade()
    finally:
        engine.dispose()


@pytest.fixture
def mixed_metric_id_rows(postgresql: Any) -> tuple[int, UUID, UUID]:
    """Create backfilled historical and post-0036 populated identities."""
    dsn = _make_db_url(postgresql)
    normalized_metric_ids = import_module(
        "coval_bench.db.migrations.versions.20260915_0036_normalized_metric_ids"
    )
    with psycopg.connect(dsn, autocommit=True) as conn:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(normalized_metric_ids, "op", SimpleNamespace(execute=conn.execute))
            normalized_metric_ids.downgrade()
        run_row = conn.execute(
            """
            INSERT INTO benchmarks_v2.runs
              (runner_sha, dataset_id, dataset_sha256, status, scheduled_at)
            VALUES ('historical-test', 'historical-dataset', %s, 'succeeded', now())
            RETURNING id
            """,
            ("b" * 64,),
        ).fetchone()
        assert run_row is not None
        run_id = run_row[0]
        historical_id = _seed_completed_evaluation(
            conn,
            run_id,
            captured_at=datetime.now(UTC) - timedelta(minutes=5),
            sample_id="historical-row",
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(normalized_metric_ids, "op", SimpleNamespace(execute=conn.execute))
            normalized_metric_ids.upgrade()
        for table in (
            "metric_evaluations",
            "dashboard_metric_values",
            "metric_values_by_bucket",
        ):
            conn.execute(
                f"UPDATE benchmarks_v2.{table} "  # noqa: S608
                "SET metric_id = benchmarks_v2.metric_id_for_code(metric_type) "
                "WHERE metric_id IS NULL"
            )
    _apply_metric_id_enforcement(dsn)
    with psycopg.connect(dsn, autocommit=True) as conn:
        fresh_id = _seed_completed_evaluation(
            conn,
            run_id,
            captured_at=datetime.now(UTC) - timedelta(minutes=1),
            sample_id="fresh-row",
        )
    return int(run_id), historical_id, fresh_id


@pytest.fixture
def early_access_registry(postgresql: Any) -> None:
    add_models(
        postgresql,
        RegisteredModel(
            benchmark=Benchmark.STT,
            provider=EA_PROVIDER,
            model=EA_MODEL,
            collected=True,
            published=False,
        ),
    )


async def _insert_evaluation(
    postgresql: Any,
    run_id: int,
    *,
    evaluation_id: UUID | None = None,
    captured_at: datetime | None = None,
    metric_type: str = "TTFA",
    metric_version: str = "v1",
    evaluation_variant: str = "default",
    status: str = "succeeded",
    value: float | None = 120.0,
    primary_key: str = "latency",
    components: tuple[tuple[str, float, str], ...] = (),
    provider: str = "deepgram",
    model: str = "nova-3",
    sample_id: str | None = None,
    dataset_id: str = "stt-v2",
    benchmark: str = "STT",
    observation_status: str = "succeeded",
    observation_id: UUID | None = None,
) -> tuple[UUID, UUID]:
    evaluation_id = evaluation_id or uuid4()
    existing_observation_id = observation_id
    observation_id = observation_id or uuid4()
    sample_id = sample_id or f"sample-{evaluation_id}"
    dsn = _make_db_url(postgresql)
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        if existing_observation_id is not None:
            observation_exists = await (
                await conn.execute(
                    "SELECT 1 FROM benchmarks_v2.benchmark_observations WHERE id = %s",
                    (existing_observation_id,),
                )
            ).fetchone()
        else:
            observation_exists = None
        if observation_exists is None:
            await conn.execute(
                """
                INSERT INTO benchmarks_v2.benchmark_observations
                  (id, run_id, dataset_id, sample_id, provider, model, voice, benchmark,
                   captured_at, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'voice-a', %s, %s, 'succeeded')
                """,
                (
                    observation_id,
                    run_id,
                    dataset_id,
                    sample_id,
                    provider,
                    model,
                    benchmark,
                    captured_at or datetime.now(UTC),
                ),
            )
            if observation_status != "succeeded":
                await conn.execute(
                    "UPDATE benchmarks_v2.benchmark_observations SET status = %s WHERE id = %s",
                    (observation_status, observation_id),
                )
        await conn.execute(
            """
            INSERT INTO benchmarks_v2.metric_evaluations
              (id, observation_id, metric_type, metric_version, evaluation_variant, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                evaluation_id,
                observation_id,
                metric_type,
                metric_version,
                evaluation_variant,
                "queued",
            ),
        )
        if status == "queued":
            return evaluation_id, observation_id
        await conn.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET status = 'running' WHERE id = %s",
            (evaluation_id,),
        )
        if status == "running":
            return evaluation_id, observation_id
        if status == "succeeded" and value is not None:
            await conn.execute(
                """
                INSERT INTO benchmarks_v2.metric_values
                  (metric_evaluation_id, value_key, unit, value, value_role)
                VALUES (%s, %s, 'milliseconds', %s, 'primary')
                """,
                (evaluation_id, primary_key, value),
            )
        if status == "succeeded":
            for key, component_value, unit in components:
                await conn.execute(
                    """
                    INSERT INTO benchmarks_v2.metric_values
                      (metric_evaluation_id, value_key, unit, value, value_role)
                    VALUES (%s, %s, %s, %s, 'component')
                    """,
                    (evaluation_id, key, unit, component_value),
                )
        await conn.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET status = %s WHERE id = %s",
            (status, evaluation_id),
        )
    return evaluation_id, observation_id


@pytest.mark.asyncio
async def test_primary_role_and_components_preserve_stored_values(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql)
    evaluation_id, _observation_id = await _insert_evaluation(
        postgresql,
        run_id,
        primary_key="primary-key-without-literal-name",
        components=(("primary", 9, "seconds"), ("leading_silence", 20, "milliseconds")),
    )
    response = await client.get(
        "/v2/results", params={"run_id": run_id, "include_components": "true"}
    )
    assert response.status_code == 200
    row = response.json()["results"][0]
    assert set(row) == {
        "captured_at",
        "provider",
        "model",
        "voice",
        "benchmark",
        "dataset_id",
        "metric_type",
        "metric_version",
        "value",
        "unit",
        "components",
    }
    assert row["value"] == 120
    assert row["unit"] == "milliseconds"
    assert row["metric_type"] == "TTFA"
    async with await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True
    ) as conn:
        assert await (
            await conn.execute(
                """
                    SELECT e.metric_id = m.id
                    FROM benchmarks_v2.metric_evaluations e
                    JOIN benchmarks_v2.metrics m ON m.code = 'TTFA'
                    WHERE e.id = %s
                    """,
                (evaluation_id,),
            )
        ).fetchone() == (True,)
    assert row["components"] == {
        "primary": {"value": 9, "unit": "seconds"},
        "leading_silence": {"value": 20, "unit": "milliseconds"},
    }


@pytest.mark.asyncio
async def test_missing_primary_is_excluded_and_does_not_use_component(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_evaluation(postgresql, run_id, value=None, components=(("component", 3, "ms"),))
    response = await client.get("/v2/results", params={"run_id": run_id})
    assert response.status_code == 200
    assert response.json()["results"] == []

    # The newest successful evaluation has no primary; it must not consume a slot.
    now = datetime.now(UTC)
    for minutes, value in ((1, 11), (2, 22)):
        await _insert_evaluation(
            postgresql, run_id, captured_at=now - timedelta(minutes=minutes), value=value
        )
    first = await client.get("/v2/results", params={"run_id": run_id, "limit": 1})
    assert [row["value"] for row in first.json()["results"]] == [11]
    assert first.json()["next_cursor"]
    second = await client.get(
        "/v2/results",
        params={"run_id": run_id, "limit": 1, "cursor": first.json()["next_cursor"]},
    )
    assert [row["value"] for row in second.json()["results"]] == [22]
    assert second.json()["next_cursor"] is None


@pytest.mark.asyncio
async def test_pagination_counts_evaluations_and_preserves_order(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql)
    now = datetime.now(UTC)
    for index in range(3):
        await _insert_evaluation(postgresql, run_id, captured_at=now - timedelta(minutes=index))
    first = await client.get("/v2/results", params={"run_id": run_id, "limit": 2})
    assert first.status_code == 200
    body = first.json()
    assert len(body["results"]) == 2
    assert body["next_cursor"]
    second = await client.get(
        "/v2/results", params={"run_id": run_id, "limit": 2, "cursor": body["next_cursor"]}
    )
    assert second.status_code == 200
    assert len(second.json()["results"]) == 1
    assert second.json()["next_cursor"] is None
    assert body["results"][0]["captured_at"] > body["results"][1]["captured_at"]


@pytest.mark.asyncio
async def test_status_and_version_filters(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql, status="partial")
    await _insert_evaluation(postgresql, run_id, metric_version="v1")
    await _insert_evaluation(postgresql, run_id, metric_version="v2", evaluation_variant="alt")
    response = await client.get(
        "/v2/results",
        params={"run_id": run_id, "metric_version": "v2", "evaluation_variant": "alt"},
    )
    assert response.status_code == 422
    default = await client.get("/v2/results", params={"run_id": run_id})
    assert default.status_code == 200
    assert len(default.json()["results"]) == 1


@pytest.mark.asyncio
async def test_default_window_only_applies_cross_run(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_evaluation(postgresql, run_id, captured_at=datetime.now(UTC) - timedelta(days=30))
    assert (await client.get("/v2/results", params={"run_id": run_id})).json()["results"]
    assert (await client.get("/v2/results")).json()["results"] == []


@pytest.mark.asyncio
async def test_components_do_not_change_page_membership(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_evaluation(postgresql, run_id, components=(("detail", 2, "ms"),))
    await _insert_evaluation(postgresql, run_id)

    plain = await client.get("/v2/results", params={"run_id": run_id, "limit": 1})
    detailed = await client.get(
        "/v2/results",
        params={"run_id": run_id, "limit": 1, "include_components": "true"},
    )
    assert plain.status_code == detailed.status_code == 200
    assert plain.json()["results"] == [
        {key: value for key, value in row.items() if key != "components"}
        for row in detailed.json()["results"]
    ]
    assert detailed.json()["results"][0]["components"] == {}


@pytest.mark.asyncio
async def test_cursor_rejects_changed_time_selector(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_evaluation(postgresql, run_id)
    await _insert_evaluation(
        postgresql, run_id, captured_at=datetime.now(UTC) - timedelta(minutes=1)
    )
    first = await client.get("/v2/results", params={"run_id": run_id, "limit": 1})
    cursor = first.json()["next_cursor"]
    assert cursor

    changed_window = await client.get(
        "/v2/results", params={"run_id": run_id, "limit": 1, "cursor": cursor, "window": "24h"}
    )
    changed_bounds = await client.get(
        "/v2/results",
        params={
            "run_id": run_id,
            "limit": 1,
            "cursor": cursor,
            "since": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        },
    )
    assert changed_window.status_code == changed_bounds.status_code == 400


@pytest.mark.asyncio
async def test_evaluation_and_run_status_filters_are_independent(
    client: AsyncClient, postgresql: Any
) -> None:
    succeeded_run = await _insert_run(postgresql, status="succeeded")
    running_run = await _insert_run(postgresql, status="running")
    await _insert_evaluation(postgresql, succeeded_run, status="failed", value=None)
    await _insert_evaluation(postgresql, running_run, status="running", value=None)

    failed_eval = await client.get(
        "/v2/results", params={"evaluation_status": "failed", "run_status": "succeeded"}
    )
    running_both = await client.get(
        "/v2/results", params={"evaluation_status": "running", "run_status": "running"}
    )
    assert failed_eval.status_code == running_both.status_code == 422


@pytest.mark.asyncio
async def test_failed_observation_evaluation_is_eligible(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql, status="partial")
    await _insert_evaluation(
        postgresql,
        run_id,
        observation_status="failed",
    )
    response = await client.get(
        "/v2/results",
        params={"run_status": "partial"},
    )
    assert response.status_code == 200
    assert [row["value"] for row in response.json()["results"]] == [120]


@pytest.mark.asyncio
async def test_newer_ineligible_evaluation_does_not_consume_page_slot(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql, status="succeeded")
    await _insert_evaluation(
        postgresql,
        run_id,
        captured_at=datetime.now(UTC) - timedelta(minutes=1),
        status="succeeded",
    )
    await _insert_evaluation(
        postgresql,
        run_id,
        captured_at=datetime.now(UTC),
        status="failed",
        value=None,
    )
    response = await client.get("/v2/results", params={"run_id": run_id, "limit": 1})
    assert response.status_code == 200
    assert [row["metric_type"] for row in response.json()["results"]] == ["TTFA"]
    assert response.json()["next_cursor"] is None


@pytest.mark.asyncio
async def test_explicit_time_bounds_are_half_open(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    since = datetime.now(UTC) - timedelta(hours=2)
    until = datetime.now(UTC) - timedelta(hours=1)
    await _insert_evaluation(postgresql, run_id, captured_at=since)
    await _insert_evaluation(postgresql, run_id, captured_at=until)
    response = await client.get(
        "/v2/results",
        params={"since": since.isoformat(), "until": until.isoformat()},
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


@pytest.mark.asyncio
async def test_openapi_marks_nullable_values_and_optional_components(
    client: AsyncClient,
) -> None:
    document = (await client.get("/openapi.json")).json()
    result_schema = document["components"]["schemas"]["ResultV2Out"]
    response_schema = document["components"]["schemas"]["ResultsV2Response"]
    assert "value" in result_schema["required"] and "unit" in result_schema["required"]
    assert "components" not in result_schema["required"]
    assert result_schema["properties"]["value"]["type"] == "number"
    assert result_schema["properties"]["components"]["type"] == "object"
    assert "next_cursor" in response_schema["required"]
    params = {item["name"]: item for item in document["paths"]["/v2/results"]["get"]["parameters"]}
    assert params["evaluation_status"]["schema"]["default"] == "succeeded"


@pytest.mark.asyncio
async def test_missing_or_invalid_cursor_key_fails_before_pool_dependency() -> None:
    def unexpected_connection() -> None:
        pytest.fail("A missing cursor key must fail before any database query")

    for key in (None, SecretStr("not-a-fernet-key")):
        settings = Settings(_env_file=None, results_cursor_key=key, posthog_disabled=True)
        app = create_app(settings=settings)
        app.state.settings = settings
        app.state.posthog = None
        app.state.pool = SimpleNamespace(connection=unexpected_connection)
        app.state.limiter.enabled = False
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for params in ({}, {"cursor": "v1.invalid"}):
                response = await client.get("/v2/results", params=params)
                assert response.status_code == 503
                assert response.json() == {"detail": "results pagination is unavailable"}


@pytest.mark.asyncio
async def test_same_observation_ties_and_cursor_are_stable_across_detail_and_limit(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql)
    captured_at = datetime.now(UTC)
    _first_id, observation_id = await _insert_evaluation(
        postgresql,
        run_id,
        evaluation_id=UUID("00000000-0000-4000-8000-000000000001"),
        captured_at=captured_at,
        metric_type="TTFA",
        metric_version="v1",
        components=(("first_component", 1, "ms"),),
    )
    _second_id, _ = await _insert_evaluation(
        postgresql,
        run_id,
        evaluation_id=UUID("00000000-0000-4000-8000-000000000004"),
        observation_id=observation_id,
        metric_type="TTFA",
        metric_version="v2",
        evaluation_variant="alt",
        components=(("second_component", 2, "ms"),),
    )
    _third_id, second_observation_id = await _insert_evaluation(
        postgresql,
        run_id,
        evaluation_id=UUID("00000000-0000-4000-8000-000000000003"),
        captured_at=captured_at,
        metric_type="WER",
        components=(("third_component", 3, "percent"),),
    )
    _fourth_id, _ = await _insert_evaluation(
        postgresql,
        run_id,
        evaluation_id=UUID("00000000-0000-4000-8000-000000000002"),
        captured_at=captured_at,
        observation_id=second_observation_id,
        metric_type="TTFT",
        components=(("fourth_component", 4, "seconds"),),
    )
    plain = await client.get("/v2/results", params={"run_id": run_id, "limit": 3})
    detailed = await client.get(
        "/v2/results",
        params={"run_id": run_id, "limit": 3, "include_components": "true"},
    )
    assert plain.status_code == detailed.status_code == 200
    assert [row["metric_type"] for row in plain.json()["results"]] == ["WER", "TTFT", "TTFA"]
    assert [row["metric_type"] for row in plain.json()["results"]] == [
        row["metric_type"] for row in detailed.json()["results"]
    ]
    assert plain.json()["next_cursor"] == detailed.json()["next_cursor"]

    first_page = await client.get("/v2/results", params={"run_id": run_id, "limit": 2})
    assert [row["metric_type"] for row in first_page.json()["results"]] == ["WER", "TTFT"]
    second_page = await client.get(
        "/v2/results",
        params={
            "run_id": run_id,
            "limit": 1,
            "include_components": "true",
            "cursor": first_page.json()["next_cursor"],
        },
    )
    assert second_page.status_code == 200
    second_row = second_page.json()["results"][0]
    assert second_row["metric_type"] == "TTFA"
    assert second_row["components"] == {"first_component": {"value": 1, "unit": "ms"}}
    assert second_page.json()["next_cursor"] is None

    empty = await client.get("/v2/results", params={"run_id": 999999, "limit": 2})
    assert empty.status_code == 200
    assert empty.json() == {"results": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_backfilled_historical_and_fresh_metric_ids_are_read_by_catalog_id(
    mixed_metric_id_rows: tuple[int, UUID, UUID], client: AsyncClient, postgresql: Any
) -> None:
    run_id, historical_id, fresh_id = mixed_metric_id_rows

    first = await client.get(
        "/v2/results", params={"run_id": run_id, "metric_type": "WER", "limit": 1}
    )
    assert first.status_code == 200
    assert [row["metric_type"] for row in first.json()["results"]] == ["WER"]
    assert first.json()["next_cursor"]
    second = await client.get(
        "/v2/results",
        params={
            "run_id": run_id,
            "metric_type": "WER",
            "limit": 1,
            "cursor": first.json()["next_cursor"],
        },
    )
    assert second.status_code == 200
    assert [row["metric_type"] for row in second.json()["results"]] == ["WER"]
    assert second.json()["results"][0]["metric_type"] == "WER"
    assert second.json()["next_cursor"] is None

    filtered = await client.get("/v2/results", params={"run_id": run_id, "metric_type": "WER"})
    assert len(filtered.json()["results"]) == 2
    async with await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True
    ) as conn:
        identities = await (
            await conn.execute(
                """
                SELECT id, metric_id
                FROM benchmarks_v2.metric_evaluations
                WHERE id IN (%s, %s)
                ORDER BY id
                """,
                (historical_id, fresh_id),
            )
        ).fetchall()
        catalog_row = await (
            await conn.execute("SELECT id FROM benchmarks_v2.metrics WHERE code='WER'")
        ).fetchone()
        assert catalog_row is not None
        catalog_id = catalog_row[0]
    by_id = {row[0]: row[1] for row in identities}
    assert by_id[historical_id] == catalog_id
    assert by_id[fresh_id] == catalog_id
    async with await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True
    ) as conn:
        for table in (
            "metric_evaluations",
            "dashboard_metric_values",
            "metric_values_by_bucket",
        ):
            row = await (
                await conn.execute(
                    f"SELECT count(*) FROM benchmarks_v2.{table} WHERE metric_id IS NULL"  # noqa: S608
                )
            ).fetchone()
            assert row == (0,)


@pytest.mark.asyncio
async def test_composed_filters_and_omitted_versions_return_expected_rows(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql, status="partial")
    _target_id, observation_id = await _insert_evaluation(
        postgresql,
        run_id,
        provider="filter-provider",
        model="filter-model",
        dataset_id="filter-dataset",
        benchmark="TTS",
        metric_type="WER",
        metric_version="v2",
    )
    await _insert_evaluation(
        postgresql,
        run_id,
        observation_id=observation_id,
        provider="filter-provider",
        model="filter-model",
        dataset_id="filter-dataset",
        benchmark="TTS",
        metric_type="WER",
        metric_version="v1",
        evaluation_variant="default",
    )
    await _insert_evaluation(postgresql, run_id, provider="other-provider")

    exact = await client.get(
        "/v2/results",
        params={
            "run_id": run_id,
            "provider": "filter-provider",
            "model": "filter-model",
            "dataset": "filter-dataset",
            "benchmark": "TTS",
            "metric_type": "WER",
            "metric_version": "v2",
            "evaluation_variant": "default",
        },
    )
    all_versions = await client.get(
        "/v2/results",
        params={
            "run_id": run_id,
            "provider": "filter-provider",
            "model": "filter-model",
            "dataset": "filter-dataset",
            "benchmark": "TTS",
            "metric_type": "WER",
        },
    )
    assert exact.status_code == 200
    assert [row["metric_version"] for row in exact.json()["results"]] == ["v2"]
    assert all_versions.status_code == 200
    assert {row["metric_version"] for row in all_versions.json()["results"]} == {"v1", "v2"}


@pytest.mark.asyncio
async def test_default_rows_and_parent_status_filters(client: AsyncClient, postgresql: Any) -> None:
    for status, value in (("succeeded", 10), ("partial", 20), ("running", 30), ("failed", 40)):
        run_id = await _insert_run(postgresql, status=status)
        await _insert_evaluation(postgresql, run_id, value=value)
        await _insert_evaluation(postgresql, run_id, status="failed", value=None)
        await _insert_evaluation(postgresql, run_id, evaluation_variant="alternate", value=99)
    default = await client.get("/v2/results")
    assert default.status_code == 200
    assert {row["value"] for row in default.json()["results"]} == {10, 20}
    for status, value in (("succeeded", 10), ("partial", 20)):
        filtered = await client.get("/v2/results", params={"run_status": status})
        assert filtered.status_code == 200
        assert [row["value"] for row in filtered.json()["results"]] == [value]


@pytest.mark.asyncio
async def test_all_status_filters_include_every_independent_state_and_ignore_legacy(
    client: AsyncClient, postgresql: Any
) -> None:
    cases = (
        ("succeeded", "succeeded"),
        ("partial", "running"),
        ("running", "failed"),
        ("failed", "queued"),
    )
    run_ids = []
    for run_status, evaluation_status in cases:
        run_id = await _insert_run(postgresql, status=run_status)
        run_ids.append(run_id)
        await _insert_evaluation(postgresql, run_id, status=evaluation_status, value=None)
    await _insert_result(postgresql, run_ids[0], provider="legacy-only", model="legacy")

    response = await client.get(
        "/v2/results", params={"evaluation_status": "all", "run_status": "all"}
    )
    assert response.status_code == 422


@pytest.mark.usefixtures("early_access_registry")
@pytest.mark.asyncio
async def test_v2_visibility_is_before_limit_and_cursor_scope_changes_are_rejected(
    app: Any, client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_evaluation(
        postgresql,
        run_id,
        provider=EA_PROVIDER,
        model=EA_MODEL,
        captured_at=datetime.now(UTC),
    )
    await _insert_evaluation(
        postgresql,
        run_id,
        provider="seed",
        model="stt",
        captured_at=datetime.now(UTC) - timedelta(minutes=1),
    )

    public = await client.get("/v2/results", params={"run_id": run_id, "limit": 1})
    partner = await client.get(
        "/v2/results", params={"run_id": run_id, "limit": 1}, headers=bearer(org_id=EA_ORG)
    )
    internal = await client.get(
        "/v2/results", params={"run_id": run_id, "limit": 1}, headers=bearer(org_id=COVAL_ORG)
    )
    assert public.status_code == partner.status_code == internal.status_code == 200
    assert public.json()["results"][0]["provider"] == "seed"
    assert partner.json()["results"][0]["provider"] == EA_PROVIDER
    assert internal.json()["results"][0]["provider"] == EA_PROVIDER
    assert "Authorization" in public.headers["Vary"]
    assert public.headers.get("Cache-Control") is None
    assert partner.headers["Cache-Control"] == "private, no-store"

    invalid_proof = await client.get(
        "/v2/results",
        params={"run_id": run_id},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert invalid_proof.status_code == 200
    assert [row["provider"] for row in invalid_proof.json()["results"]] == ["seed"]
    hidden_filter = await client.get(
        "/v2/results", params={"run_id": run_id, "provider": EA_PROVIDER, "model": EA_MODEL}
    )
    assert hidden_filter.status_code == 200
    assert hidden_filter.json()["results"] == []

    cursor = partner.json()["next_cursor"]
    assert cursor
    app.state.settings.clerk_org_exclusive = json.dumps({EA_ORG: f"{EA_PROVIDER}/{EA_MODEL}"})
    changed_visibility = await client.get(
        "/v2/results",
        params={"run_id": run_id, "limit": 1, "cursor": cursor},
        headers=bearer(org_id=EA_ORG),
    )
    assert changed_visibility.status_code == 400
    exclusive = await client.get(
        "/v2/results", params={"run_id": run_id}, headers=bearer(org_id=EA_ORG)
    )
    assert [row["provider"] for row in exclusive.json()["results"]] == [EA_PROVIDER]


@pytest.mark.asyncio
async def test_cursor_validation_time_conflicts_and_frozen_bounds(
    client: AsyncClient, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from coval_bench.api.routers import results_v2 as results_router

    run_id = await _insert_run(postgresql)
    now = datetime.now(UTC).replace(microsecond=0)
    await _insert_evaluation(postgresql, run_id, captured_at=now - timedelta(minutes=1))
    await _insert_evaluation(postgresql, run_id, captured_at=now - timedelta(minutes=2))
    first = await client.get(
        "/v2/results",
        params={
            "since": (now - timedelta(hours=1)).isoformat(),
            "until": (now + timedelta(hours=1)).isoformat(),
            "limit": 1,
        },
    )
    cursor = first.json()["next_cursor"]
    assert cursor

    changed_until = await client.get(
        "/v2/results",
        params={
            "since": (now - timedelta(hours=1)).isoformat(),
            "until": (now + timedelta(hours=2)).isoformat(),
            "limit": 1,
            "cursor": cursor,
        },
    )
    contradictory = await client.get(
        "/v2/results",
        params={"window": "24h", "since": now.isoformat(), "cursor": cursor},
    )
    naive = await client.get("/v2/results", params={"since": "2026-09-15T00:00:00"})
    assert changed_until.status_code == contradictory.status_code == naive.status_code == 400

    relative_first = await client.get("/v2/results", params={"window": "24h", "limit": 1})
    relative_cursor = relative_first.json()["next_cursor"]
    assert relative_cursor

    cursor_key = SecretStr("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    raw = decode_cursor(relative_cursor, cursor_key)
    raw.pop("until")
    changed_until = await client.get(
        "/v2/results",
        params={
            "window": "24h",
            "limit": 1,
            "cursor": encode_cursor(raw, cursor_key),
        },
    )
    malformed = await client.get("/v2/results", params={"cursor": "not-base64"})
    deep = dict(raw, extra=[[[]]])
    deep_cursor = encode_cursor(deep, cursor_key)
    deep_response = await client.get("/v2/results", params={"cursor": deep_cursor})
    assert changed_until.status_code == malformed.status_code == deep_response.status_code == 400
    for field, value in (("until", 1), ("anchor_id", 1), ("v", True)):
        invalid_payload = dict(decode_cursor(relative_cursor, cursor_key))
        invalid_payload[field] = value
        invalid = await client.get(
            "/v2/results", params={"cursor": encode_cursor(invalid_payload, cursor_key)}
        )
        assert invalid.status_code == 400

    await _insert_evaluation(postgresql, run_id, captured_at=now - timedelta(minutes=3))
    await _insert_evaluation(postgresql, run_id, captured_at=now + timedelta(minutes=1))
    await _insert_evaluation(postgresql, run_id, captured_at=now - timedelta(hours=47))

    class LaterDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> LaterDateTime:
            return cls.fromtimestamp((now + timedelta(days=2)).timestamp(), tz=tz)

    monkeypatch.setattr(results_router, "datetime", LaterDateTime)
    continued = await client.get(
        "/v2/results", params={"window": "24h", "limit": 100, "cursor": relative_cursor}
    )
    assert continued.status_code == 200
    continued_times = [row["captured_at"] for row in continued.json()["results"]]
    assert len(continued_times) == 2
    assert all(time < (now + timedelta(minutes=1)).isoformat() for time in continued_times)
