# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the S2S V2V fetch job."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from coval_bench.config import Settings
from coval_bench.db.models import ResultStatus, Run, RunStatus
from coval_bench.s2s import fetch_v2v
from coval_bench.s2s.fetch_v2v import AgentSpec

SPEC = AgentSpec(agent_id_attr="coval_s2s_openai_agent_id", provider="openai", model="gpt-realtime")
SLOT = datetime(2026, 7, 7, tzinfo=UTC)
_LIST_ONE: dict[str, Any] = {"runs": [{"run_id": "R1", "status": "COMPLETED"}]}


def _fake_client(
    list_json: dict[str, Any],
    run_json: dict[str, Any],
    captured: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    """AsyncClient that answers /runs (list) and /runs/{id} (get) from fixtures."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        if request.url.path.endswith("/runs"):
            return httpx.Response(200, json=list_json)
        return httpx.Response(200, json=run_json)

    return httpx.AsyncClient(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))


def _run_json(values: list[dict[str, Any]], metric_id: str = "MID") -> dict[str, Any]:
    return {"run": {"results": {"metrics": {metric_id: {"values": values}}}}}


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


@pytest.mark.asyncio
async def test_per_clip_rows_maps_values() -> None:
    values = [
        {"simulation_output_id": "s1", "value": 0.842},
        {"simulation_output_id": "s2", "value": None},
        {"value": 0.5},  # missing sim id -> index fallback in the clip key
    ]
    async with _fake_client(_LIST_ONE, _run_json(values)) as client:
        rows = await fetch_v2v.per_clip_rows(
            client, run_pk=1, coval_run_id="R1", metric_id="MID", spec=SPEC
        )
    assert [r.metric_value for r in rows] == [842.0, None, 500.0]
    assert [r.status for r in rows] == [
        ResultStatus.SUCCESS,
        ResultStatus.FAILED,
        ResultStatus.SUCCESS,
    ]
    assert [r.audio_filename for r in rows] == ["R1/s1", "R1/s2", "R1/2"]
    assert all(r.benchmark == "S2S" and r.metric_type == "V2V" for r in rows)


@pytest.mark.asyncio
async def test_latest_completed_run_id() -> None:
    captured: list[httpx.Request] = []
    async with _fake_client(_LIST_ONE, {}, captured) as client:
        assert await fetch_v2v.latest_completed_run_id(client, "a1") == "R1"
    filter_expr = captured[0].url.params["filter"]
    assert 'status="COMPLETED"' in filter_expr
    assert 'agent_id="a1"' in filter_expr

    async with _fake_client({"runs": []}, {}) as client:
        assert await fetch_v2v.latest_completed_run_id(client, "a1") is None


@pytest.mark.asyncio
async def test_fetch_one_provider_succeeded() -> None:
    writer = _stub_writer()
    values = [{"simulation_output_id": f"s{i}", "value": 0.5} for i in range(3)]
    async with _fake_client(_LIST_ONE, _run_json(values)) as client:
        status = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_id="MID",
            runner_sha="test",
            scheduled_at=SLOT,
        )
    assert status is RunStatus.SUCCEEDED
    writer.record_results.assert_awaited_once()
    writer.refresh_bucket.assert_awaited_once()
    assert writer.finish_run.await_args.kwargs["status"] is RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_fetch_one_provider_partial() -> None:
    writer = _stub_writer()
    values = [
        {"simulation_output_id": "s1", "value": 0.5},
        {"simulation_output_id": "s2", "value": None},
    ]
    async with _fake_client(_LIST_ONE, _run_json(values)) as client:
        status = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_id="MID",
            runner_sha="test",
            scheduled_at=SLOT,
        )
    assert status is RunStatus.PARTIAL


@pytest.mark.asyncio
async def test_fetch_one_provider_no_run_failed() -> None:
    writer = _stub_writer()
    async with _fake_client({"runs": []}, {}) as client:
        status = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_id="MID",
            runner_sha="test",
            scheduled_at=SLOT,
        )
    assert status is RunStatus.FAILED
    writer.record_results.assert_not_awaited()
    writer.refresh_bucket.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_one_provider_skips_already_ingested_run() -> None:
    writer = _stub_writer()
    writer.coval_run_ingested = AsyncMock(return_value=True)  # run already in the DB
    async with _fake_client(_LIST_ONE, _run_json([])) as client:
        status = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_id="MID",
            runner_sha="test",
            scheduled_at=SLOT,
        )
    assert status is RunStatus.SUCCEEDED  # no-op: last good data left in place
    writer.start_run.assert_not_awaited()
    writer.record_results.assert_not_awaited()
    writer.refresh_bucket.assert_not_awaited()


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
    values = [{"simulation_output_id": f"s{i}", "value": 0.5} for i in range(2)]
    client = _fake_client(_LIST_ONE, _run_json(values))

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
