# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the S2S V2V fetch job."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from coval_bench.config import Settings
from coval_bench.db.models import ResultStatus, Run, RunStatus
from coval_bench.s2s import fetch_v2v
from coval_bench.s2s.fetch_v2v import AgentSpec, CovalRun

SPEC = AgentSpec(agent_id_attr="coval_s2s_openai_agent_id", provider="openai", model="gpt-realtime")


def _iso(age: timedelta) -> str:
    return (datetime.now(tz=UTC) - age).isoformat().replace("+00:00", "Z")


def _list_json(*runs: dict[str, Any]) -> dict[str, Any]:
    return {"runs": list(runs)}


def _run_json(
    values: list[dict[str, Any]],
    metric_id: str = "MID",
    error_status: str | None = None,
) -> dict[str, Any]:
    run: dict[str, Any] = {"results": {"metrics": {metric_id: {"values": values}}}}
    if error_status:
        run["error_status"] = error_status
    return {"run": run}


def _fake_client(
    list_json: dict[str, Any],
    run_json: dict[str, Any] | dict[str, dict[str, Any]],
    captured: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    """AsyncClient answering /runs (list) and /runs/{id} (get) from fixtures.

    ``run_json`` is either one detail fixture for every id, or a mapping of
    run id -> fixture.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        if request.url.path.endswith("/runs"):
            return httpx.Response(200, json=list_json)
        run_id = request.url.path.rsplit("/", 1)[-1]
        if "run" in run_json:
            return httpx.Response(200, json=run_json)
        return httpx.Response(200, json=run_json[run_id])

    return httpx.AsyncClient(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))


def _stub_writer() -> MagicMock:
    writer = MagicMock()
    writer.start_run = AsyncMock(
        return_value=Run(
            id=1,
            runner_sha="test",
            dataset_id="s2s-v1",
            dataset_sha256="x",
            status=RunStatus.RUNNING,
        )
    )
    writer.coval_run_ingested = AsyncMock(return_value=False)
    writer.record_results = AsyncMock()
    writer.finish_run = AsyncMock()
    writer.refresh_bucket = AsyncMock()
    writer.refresh_stats_matviews = AsyncMock()
    return writer


async def _fetch(client: httpx.AsyncClient, writer: MagicMock) -> tuple[RunStatus, int]:
    return await fetch_v2v._fetch_one_provider(
        client,
        writer,
        spec=SPEC,
        agent_id="a1",
        metric_id="MID",
        runner_sha="test",
        period_seconds=10_800,
        stale_grace_seconds=5_400,
    )


def test_bucket_start_floors_to_grid() -> None:
    at = datetime(2026, 7, 7, 4, 59, 59, tzinfo=UTC)
    assert fetch_v2v._bucket_start(at, 10_800) == datetime(2026, 7, 7, 3, tzinfo=UTC)
    on_boundary = datetime(2026, 7, 7, 9, tzinfo=UTC)
    assert fetch_v2v._bucket_start(on_boundary, 10_800) == on_boundary


def test_result_rows_maps_values() -> None:
    values: list[dict[str, Any]] = [
        {"simulation_output_id": "s1", "value": 0.842},
        {"simulation_output_id": "s2", "value": None},
        {"value": 0.5},  # missing sim id -> index fallback in the clip key
    ]
    rows = fetch_v2v._result_rows(values, run_pk=1, coval_run_id="R1", spec=SPEC)
    assert [r.metric_value for r in rows] == [842.0, None, 500.0]
    assert [r.status for r in rows] == [
        ResultStatus.SUCCESS,
        ResultStatus.FAILED,
        ResultStatus.SUCCESS,
    ]
    assert [r.audio_filename for r in rows] == ["R1/s1", "R1/s2", "R1/2"]
    assert all(r.benchmark == "S2S" and r.metric_type == "V2V" for r in rows)


@pytest.mark.asyncio
async def test_recent_completed_runs_window_and_parse() -> None:
    captured: list[httpx.Request] = []
    list_json = _list_json(
        {"run_id": "R3", "create_time": _iso(timedelta(hours=1))},
        {
            "run_id": "R2",
            "create_time": _iso(timedelta(hours=4)),
            "error_status": "EXECUTION_FAILURE",
        },
        {"run_id": "R1", "create_time": _iso(timedelta(days=3))},  # outside the window
        {"run_id": "R0"},  # no create_time -> kept
    )
    async with _fake_client(list_json, {}, captured) as client:
        runs = await fetch_v2v.recent_completed_runs(client, "a1")

    filter_expr = captured[0].url.params["filter"]
    assert 'status="COMPLETED"' in filter_expr
    assert 'agent_id="a1"' in filter_expr
    assert [r.run_id for r in runs] == ["R3", "R2", "R0"]
    assert runs[1].error_status == "EXECUTION_FAILURE"
    assert runs[0].error_status is None
    assert runs[2].create_time is None

    async with _fake_client({"runs": []}, {}) as client:
        assert await fetch_v2v.recent_completed_runs(client, "a1") == []


@pytest.mark.asyncio
async def test_ingest_run_slots_by_create_time() -> None:
    writer = _stub_writer()
    created = datetime(2026, 7, 7, 1, 15, tzinfo=UTC)
    values = [{"simulation_output_id": f"s{i}", "value": 0.5} for i in range(3)]
    async with _fake_client({}, _run_json(values)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=created, error_status=None),
            metric_id="MID",
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    assert writer.start_run.await_args.kwargs["scheduled_at"] == datetime(2026, 7, 7, 0, tzinfo=UTC)
    writer.record_results.assert_awaited_once()
    writer.refresh_bucket.assert_awaited_once()
    assert writer.finish_run.await_args.kwargs["status"] is RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_ingest_run_partial_and_failed() -> None:
    writer = _stub_writer()
    mixed: list[dict[str, Any]] = [
        {"simulation_output_id": "s1", "value": 0.5},
        {"simulation_output_id": "s2", "value": None},
    ]
    run = CovalRun(run_id="R1", create_time=None, error_status=None)
    async with _fake_client({}, _run_json(mixed)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=run,
            metric_id="MID",
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.PARTIAL

    writer = _stub_writer()
    all_null: list[dict[str, Any]] = [{"simulation_output_id": "s1", "value": None}]
    async with _fake_client({}, _run_json(all_null)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=run,
            metric_id="MID",
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.FAILED
    writer.refresh_bucket.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_run_skips_before_any_write() -> None:
    # Errored detail (reconciler zombie): skipped, no run row created.
    writer = _stub_writer()
    run = CovalRun(run_id="R1", create_time=None, error_status=None)
    async with _fake_client({}, _run_json([], error_status="EXECUTION_FAILURE")) as client:
        assert (
            await fetch_v2v._ingest_run(
                client,
                writer,
                spec=SPEC,
                coval_run=run,
                metric_id="MID",
                runner_sha="test",
                period_seconds=10_800,
            )
            is None
        )
    writer.start_run.assert_not_awaited()

    # Metric absent: skipped, no run row created.
    writer = _stub_writer()
    async with _fake_client({}, _run_json([], metric_id="OTHER")) as client:
        assert (
            await fetch_v2v._ingest_run(
                client,
                writer,
                spec=SPEC,
                coval_run=run,
                metric_id="MID",
                runner_sha="test",
                period_seconds=10_800,
            )
            is None
        )
    writer.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_one_provider_ingests_every_new_run() -> None:
    writer = _stub_writer()
    writer.start_run = AsyncMock(
        side_effect=[
            Run(
                id=i,
                runner_sha="t",
                dataset_id="s2s-v1",
                dataset_sha256="x",
                status=RunStatus.RUNNING,
            )
            for i in (1, 2)
        ]
    )
    list_json = _list_json(
        {"run_id": "R2", "create_time": _iso(timedelta(hours=1))},
        {"run_id": "R1", "create_time": _iso(timedelta(hours=4))},
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    async with _fake_client(list_json, _run_json(values)) as client:
        status, ingested = await _fetch(client, writer)

    assert (status, ingested) == (RunStatus.SUCCEEDED, 2)
    assert writer.start_run.await_count == 2
    written = [c.args[0][0].audio_filename for c in writer.record_results.await_args_list]
    assert written == ["R2/s1", "R1/s1"]


@pytest.mark.asyncio
async def test_fetch_one_provider_noop_when_fresh() -> None:
    writer = _stub_writer()
    writer.coval_run_ingested = AsyncMock(return_value=True)
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=2))})
    async with _fake_client(list_json, {}) as client:
        status, ingested = await _fetch(client, writer)
    assert (status, ingested) == (RunStatus.SUCCEEDED, 0)
    writer.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_one_provider_stale_fails() -> None:
    # Newest usable run is older than period + grace (4.5h) -> stale.
    writer = _stub_writer()
    writer.coval_run_ingested = AsyncMock(return_value=True)
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=6))})
    async with _fake_client(list_json, {}) as client:
        status, ingested = await _fetch(client, writer)
    assert (status, ingested) == (RunStatus.FAILED, 0)

    # No usable runs at all -> stale.
    writer = _stub_writer()
    async with _fake_client({"runs": []}, {}) as client:
        status, ingested = await _fetch(client, writer)
    assert (status, ingested) == (RunStatus.FAILED, 0)


@pytest.mark.asyncio
async def test_fetch_one_provider_skips_errored_ingests_older_clean() -> None:
    # Newest run errored (summary view): skip it, ingest the older clean one.
    writer = _stub_writer()
    list_json = _list_json(
        {
            "run_id": "R2",
            "create_time": _iso(timedelta(hours=1)),
            "error_status": "EXECUTION_FAILURE",
        },
        {"run_id": "R1", "create_time": _iso(timedelta(hours=4))},
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    async with _fake_client(list_json, {"R1": _run_json(values)}, None) as client:
        status, ingested = await _fetch(client, writer)

    assert (status, ingested) == (RunStatus.SUCCEEDED, 1)
    written = [c.args[0][0].audio_filename for c in writer.record_results.await_args_list]
    assert written == ["R1/s1"]


@pytest.mark.asyncio
async def test_fetch_and_write_v2v_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # gemini agent id left unset (None) -> that provider is skipped.
    monkeypatch.delenv("COVAL_S2S_GEMINI_AGENT_ID", raising=False)
    settings = Settings(
        runner_sha="test",
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
    )

    writer = _stub_writer()
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})
    values = [{"simulation_output_id": f"s{i}", "value": 0.5} for i in range(2)]
    client = _fake_client(list_json, _run_json(values))

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    monkeypatch.setattr(fetch_v2v, "_client", lambda _s: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)

    statuses = await fetch_v2v.fetch_and_write_v2v(settings)

    # only openai runs (gemini unset), and it fully succeeds.
    assert statuses == {"openai": RunStatus.SUCCEEDED}
    writer.refresh_stats_matviews.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_and_write_v2v_noop_skips_matview_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COVAL_S2S_GEMINI_AGENT_ID", raising=False)
    settings = Settings(
        runner_sha="test",
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
    )

    writer = _stub_writer()
    writer.coval_run_ingested = AsyncMock(return_value=True)  # nothing new this tick
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})
    client = _fake_client(list_json, {})

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    monkeypatch.setattr(fetch_v2v, "_client", lambda _s: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)

    statuses = await fetch_v2v.fetch_and_write_v2v(settings)

    assert statuses == {"openai": RunStatus.SUCCEEDED}
    writer.refresh_stats_matviews.assert_not_awaited()
