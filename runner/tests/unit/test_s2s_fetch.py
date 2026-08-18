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
from click.testing import CliRunner
from structlog.testing import capture_logs

from coval_bench.config import Settings
from coval_bench.db.models import ResultStatus, Run, RunStatus
from coval_bench.logging import log_run_failed, log_run_partial
from coval_bench.registries import Metric
from coval_bench.s2s import fetch_v2v
from coval_bench.s2s.conditions import (
    DATASET_ID_MULTITURN,
    DATASET_ID_MULTITURN_NOISY,
    FAMILY_HAPPYPATH,
    FAMILY_MULTITURN,
    Condition,
    condition_for,
)
from coval_bench.s2s.fetch_v2v import AgentSpec, CovalRun

IDS = {Metric.V2V: "MID", Metric.INSTRUCTION_FOLLOWING: "IID"}
LATENCY_IDS = {Metric.V2V: "MID"}
ALL_IDS = {**IDS, Metric.INTERRUPTION_RATE: "RID"}

SPEC = AgentSpec(agent_id_attr="coval_s2s_openai_agent_id", provider="openai", model="gpt-realtime")


def _iso(age: timedelta) -> str:
    return (datetime.now(tz=UTC) - age).isoformat().replace("+00:00", "Z")


def _list_json(*runs: dict[str, Any]) -> dict[str, Any]:
    return {"runs": list(runs)}


def _run_json(
    values: list[dict[str, Any]],
    metric_id: str = "MID",
    error_status: str | None = "SUCCESS",
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
    writer.coval_metric_ingested = AsyncMock(return_value=False)
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
        metric_ids=LATENCY_IDS,
        runner_sha="test",
        period_seconds=10_800,
        stale_grace_seconds=5_400,
    )


def test_expected_sample_models_excludes_non_publishing_agents() -> None:
    settings = Settings.model_construct(
        coval_s2s_openai_agent_id="a1",
        coval_s2s_gray_agent_id="a2",
        coval_s2s_red_agent_id="a3",
    )
    expected = fetch_v2v._expected_sample_models(settings)

    assert ("openai", "gpt-realtime") in expected
    assert ("colors", "gray") not in expected
    assert ("colors", "red") not in expected


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
    rows = fetch_v2v._s2s_rows(values, metric=Metric.V2V, run_pk=1, coval_run_id="R1", spec=SPEC)
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
        {"run_id": "R3", "create_time": _iso(timedelta(hours=1)), "error_status": "SUCCESS"},
        {
            "run_id": "R2",
            "create_time": _iso(timedelta(hours=4)),
            "error_status": "EXECUTION_FAILURE",
        },
        {"run_id": "R1", "create_time": _iso(timedelta(days=3))},  # outside the window
        {"run_id": "R0"},  # no create_time -> kept
    )
    async with _fake_client(list_json, {}, captured) as client:
        runs = await fetch_v2v.recent_completed_runs(client, "a1", period_seconds=10_800)

    filter_expr = captured[0].url.params["filter"]
    assert 'status="COMPLETED"' in filter_expr
    assert 'agent_id="a1"' in filter_expr
    assert [r.run_id for r in runs] == ["R3", "R2", "R0"]
    assert runs[1].error_status == "EXECUTION_FAILURE"
    assert runs[0].error_status is None
    assert runs[2].create_time is None

    async with _fake_client({"runs": []}, {}) as client:
        assert await fetch_v2v.recent_completed_runs(client, "a1", period_seconds=10_800) == []


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
            metric_ids=LATENCY_IDS,
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
            metric_ids=LATENCY_IDS,
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
            metric_ids=LATENCY_IDS,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.FAILED
    writer.refresh_bucket.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_run_skips_before_any_write() -> None:
    # Errored detail: skipped, no run row created.
    writer = _stub_writer()
    run = CovalRun(run_id="R1", create_time=None, error_status=None)
    async with _fake_client({}, _run_json([], error_status="EXECUTION_FAILURE")) as client:
        assert (
            await fetch_v2v._ingest_run(
                client,
                writer,
                spec=SPEC,
                coval_run=run,
                metric_ids=LATENCY_IDS,
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
                metric_ids=LATENCY_IDS,
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
    writer.coval_metric_ingested = AsyncMock(return_value=True)
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=2))})
    async with _fake_client(list_json, {}) as client:
        status, ingested = await _fetch(client, writer)
    assert (status, ingested) == (RunStatus.SUCCEEDED, 0)
    writer.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_one_provider_stale_fails() -> None:
    # Newest usable run is older than period + grace (4.5h) -> stale.
    writer = _stub_writer()
    writer.coval_metric_ingested = AsyncMock(return_value=True)
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
async def test_fetch_one_provider_stale_wins_over_backfill() -> None:
    # Only an old run ingests this tick: rows land, provider still FAILED.
    writer = _stub_writer()
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=6))})
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    async with _fake_client(list_json, _run_json(values)) as client:
        status, ingested = await _fetch(client, writer)
    assert (status, ingested) == (RunStatus.FAILED, 1)
    writer.record_results.assert_awaited_once()


@pytest.mark.asyncio
async def test_optional_metric_alone_does_not_prove_freshness() -> None:
    writer = _stub_writer()

    async def ingested(*, provider: str, coval_run_id: str, metric_type: Metric) -> bool:
        return metric_type is Metric.INSTRUCTION_FOLLOWING

    writer.coval_metric_ingested = AsyncMock(side_effect=ingested)
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})
    run_json = {
        "run": {
            "error_status": "SUCCESS",
            "results": {
                "metrics": {"IID": {"values": [{"simulation_output_id": "s1", "value": "YES"}]}}
            },
        }
    }
    async with _fake_client(list_json, run_json) as client:
        status, _ = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=IDS,
            test_set_id="TS1",
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
        )
    assert status is RunStatus.FAILED


@pytest.mark.asyncio
async def test_fetch_one_provider_unknown_age_is_stale() -> None:
    # A usable run without a parseable create_time is no evidence of freshness.
    writer = _stub_writer()
    writer.coval_metric_ingested = AsyncMock(return_value=True)
    list_json = _list_json({"run_id": "R1"})
    async with _fake_client(list_json, {}) as client:
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
    assert statuses == {"openai:gpt-realtime": RunStatus.SUCCEEDED}
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
    writer.coval_metric_ingested = AsyncMock(return_value=True)  # nothing new this tick
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})
    client = _fake_client(list_json, {})

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    monkeypatch.setattr(fetch_v2v, "_client", lambda _s: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)

    statuses = await fetch_v2v.fetch_and_write_v2v(settings)

    assert statuses == {"openai:gpt-realtime": RunStatus.SUCCEEDED}
    writer.refresh_stats_matviews.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_ingested_run_still_becomes_sample_candidate() -> None:
    from coval_bench.s2s.samples import SampleRun

    writer = _stub_writer()
    writer.coval_metric_ingested = AsyncMock(return_value=True)
    list_json = _list_json(
        {"run_id": "R1", "create_time": _iso(timedelta(hours=1)), "error_status": "SUCCESS"}
    )
    sampled: list[SampleRun] = []
    async with _fake_client(list_json, {}) as client:
        _status, ingested = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
        )
    assert ingested == 0
    assert [(r.provider, r.coval_run_id) for r in sampled] == [("openai", "R1")]


@pytest.mark.asyncio
async def test_embargoed_agent_never_becomes_a_sample_candidate() -> None:
    """An embargoed agent's recordings must never reach the public samples card."""
    from coval_bench.s2s.samples import SampleRun

    writer = _stub_writer()
    writer.coval_metric_ingested = AsyncMock(return_value=True)
    list_json = _list_json(
        {"run_id": "R1", "create_time": _iso(timedelta(hours=1)), "error_status": "SUCCESS"}
    )
    embargoed = AgentSpec(
        agent_id_attr="coval_s2s_gray_agent_id",
        provider="colors",
        model="gray",
        publish_samples=False,
    )
    sampled: list[SampleRun] = []
    async with _fake_client(list_json, {}) as client:
        await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=embargoed,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
        )
    assert sampled == []


@pytest.mark.asyncio
async def test_each_bucket_contributes_its_own_sample_candidate() -> None:
    """Runs in different buckets are both candidates — this is what recovers a missed day."""
    from coval_bench.s2s.samples import SampleRun

    writer = _stub_writer()
    # Exactly one period apart, so the two always floor to different buckets.
    list_json = _list_json(
        {"run_id": "R2", "create_time": _iso(timedelta(hours=1))},
        {"run_id": "R1", "create_time": _iso(timedelta(hours=4))},
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    sampled: list[SampleRun] = []
    async with _fake_client(list_json, _run_json(values)) as client:
        await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
        )
    assert [r.coval_run_id for r in sampled] == ["R2", "R1"]
    assert len({r.bucket_at for r in sampled}) == 2


@pytest.mark.asyncio
async def test_one_bucket_keeps_only_its_newest_candidate() -> None:
    """Staggered arrivals within a bucket must not shrink that bucket's sample."""
    from coval_bench.s2s.samples import SampleRun

    writer = _stub_writer()
    # Same create_time, so both are certainly in one bucket; newest-first wins.
    same_time = _iso(timedelta(hours=1))
    list_json = _list_json(
        {"run_id": "R2", "create_time": same_time},
        {"run_id": "R1", "create_time": same_time},
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    sampled: list[SampleRun] = []
    async with _fake_client(list_json, _run_json(values)) as client:
        await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
        )
    assert [r.coval_run_id for r in sampled] == ["R2"]


@pytest.mark.asyncio
async def test_stale_provider_lends_no_sample_candidate() -> None:
    from coval_bench.s2s.samples import SampleRun

    writer = _stub_writer()
    writer.coval_metric_ingested = AsyncMock(return_value=True)
    list_json = _list_json(
        {"run_id": "R1", "create_time": _iso(timedelta(hours=5)), "error_status": "SUCCESS"}
    )
    sampled: list[SampleRun] = []
    async with _fake_client(list_json, {}) as client:
        status, _ingested = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
        )
    assert status is RunStatus.FAILED
    assert sampled == []


def test_instruction_verdict_classifies() -> None:
    # The binary judge only ever emits canonical YES / NO / UNKNOWN (BinaryResult.parse
    # normalizes case and maps anything missing/invalid to UNKNOWN server-side).
    assert fetch_v2v._instruction_verdict("YES") is True
    assert fetch_v2v._instruction_verdict("NO") is False
    assert fetch_v2v._instruction_verdict("UNKNOWN") is None


def test_instruction_rows_maps_verdicts() -> None:
    values: list[dict[str, Any]] = [
        {"simulation_output_id": "s1", "value": "YES"},
        {"simulation_output_id": "s2", "value": "NO"},
        {"simulation_output_id": "s3", "value": "UNKNOWN"},
    ]
    rows = fetch_v2v._s2s_rows(
        values, metric=Metric.INSTRUCTION_FOLLOWING, run_pk=1, coval_run_id="R1", spec=SPEC
    )
    # UNKNOWN produces no row (excluded from the pool); only YES/NO are written.
    assert [r.metric_value for r in rows] == [100.0, 0.0]
    assert [r.status for r in rows] == [ResultStatus.SUCCESS, ResultStatus.SUCCESS]
    assert [r.audio_filename for r in rows] == ["R1/s1", "R1/s2"]
    assert all(
        r.metric_type == Metric.INSTRUCTION_FOLLOWING and r.metric_units == "percent" for r in rows
    )


def test_instruction_verdict_raises_on_unexpected() -> None:
    with pytest.raises(fetch_v2v.InvalidInstructionVerdict):
        fetch_v2v._instruction_verdict("MAYBE")
    with pytest.raises(fetch_v2v.InvalidInstructionVerdict):
        fetch_v2v._instruction_verdict(None)


def test_population_mismatch() -> None:
    anchor = [{"simulation_output_id": "s1"}, {"simulation_output_id": "s2"}]
    assert fetch_v2v._population_mismatch(anchor, anchor) is None
    # different population (same count) -> diff reported
    other = [{"simulation_output_id": "s1"}, {"simulation_output_id": "s3"}]
    assert fetch_v2v._population_mismatch(anchor, other) == {
        "missing_ids": ["s2"],
        "extra_ids": ["s3"],
        "duplicate_ids": False,
    }
    dup = [{"simulation_output_id": "s1"}, {"simulation_output_id": "s1"}]
    dup_diff = fetch_v2v._population_mismatch(anchor, dup)
    assert dup_diff is not None
    assert dup_diff["duplicate_ids"] is True


def test_has_duplicate_ids_guards_the_anchor() -> None:
    # The anchor has nothing to be compared against, so this is its only check.
    assert not fetch_v2v._has_duplicate_ids([{"simulation_output_id": "s1"}])
    assert fetch_v2v._has_duplicate_ids(
        [{"simulation_output_id": "s1"}, {"simulation_output_id": "s1"}]
    )


def test_dataset_identity() -> None:
    assert fetch_v2v._dataset_identity("TS1") == ("s2s-multiturn-v1", "TS1")
    single_turn = fetch_v2v._dataset_identity(None)
    assert single_turn is not None
    assert single_turn[0] == "s2s-v1"


def test_dataset_identity_splits_the_noisy_caller() -> None:
    """The pre-map setting still splits the noisy persona while the map is unset."""
    assert fetch_v2v._dataset_identity("TS1", "PN", "PN") == (
        "s2s-multiturn-noisy-v1",
        "TS1:PN",
    )
    # Every other persona in that test set stays pooled as the clean condition.
    assert fetch_v2v._dataset_identity("TS1", "PCLEAN", "PN") == ("s2s-multiturn-v1", "TS1")
    # Unset setting: no persona is noisy, so nothing splits.
    assert fetch_v2v._dataset_identity("TS1", "PN", None) == ("s2s-multiturn-v1", "TS1")
    # Single-turn has no test set, so it keeps the packaged manifest.
    single_turn = fetch_v2v._dataset_identity(None, "PN", "PN")
    assert single_turn is not None
    assert single_turn[0] == "s2s-v1"


def test_dataset_identity_keeps_families_apart() -> None:
    """The same persona lands in a different dataset per test-set family."""
    personas = {"PC": Condition.CLEAN, "PN": Condition.NOISY, "PA": Condition.ACCENTED}
    assert fetch_v2v._dataset_identity(
        "TS1", "PC", family=FAMILY_MULTITURN, persona_conditions=personas
    ) == ("s2s-multiturn-v1", "TS1")
    assert fetch_v2v._dataset_identity(
        "TS2", "PC", family=FAMILY_HAPPYPATH, persona_conditions=personas
    ) == ("s2s-happypath-v1", "TS2")
    assert fetch_v2v._dataset_identity(
        "TS2", "PN", family=FAMILY_HAPPYPATH, persona_conditions=personas
    ) == ("s2s-happypath-noisy-v1", "TS2:PN")
    assert fetch_v2v._dataset_identity(
        "TS2", "PA", family=FAMILY_HAPPYPATH, persona_conditions=personas
    ) == ("s2s-happypath-accented-v1", "TS2:PA")


def test_dataset_identity_skips_known_and_faults_unknown_personas() -> None:
    personas = {"PSKIP": Condition.SKIP}
    # Known but deliberately not ingested.
    assert fetch_v2v._dataset_identity("TS1", "PSKIP", persona_conditions=personas) is None
    # Unmapped: loud, because counting it as clean would be invisible.
    with pytest.raises(ValueError, match="no condition mapped"):
        fetch_v2v._dataset_identity("TS1", "PUNKNOWN", persona_conditions=personas)


def test_dataset_identity_map_supersedes_the_noisy_setting() -> None:
    """Once the map is set the single-persona setting no longer applies."""
    assert fetch_v2v._dataset_identity(
        "TS1", "PN", "PN", persona_conditions={"PN": Condition.CLEAN}
    ) == ("s2s-multiturn-v1", "TS1")


def test_dataset_identity_rejects_a_condition_its_family_lacks() -> None:
    """A persona mapped to a condition the shared set never runs is a config error."""
    with pytest.raises(ValueError, match="no dataset id"):
        fetch_v2v._dataset_identity(
            "TS1", "PA", family=FAMILY_MULTITURN, persona_conditions={"PA": Condition.ACCENTED}
        )


@pytest.mark.asyncio
async def test_noisy_and_clean_runs_land_in_different_datasets() -> None:
    """One scan, two personas: the dataset is chosen per run, not per tick."""
    writer = _stub_writer()
    list_json = _list_json(
        {"run_id": "RNOISY", "create_time": _iso(timedelta(hours=1)), "persona_id": "PN"},
        {"run_id": "RCLEAN", "create_time": _iso(timedelta(hours=1)), "persona_id": "PCLEAN"},
    )
    latency = [{"simulation_output_id": "s1", "value": 0.5}]
    instruction = [{"simulation_output_id": "s1", "value": "YES"}]
    async with _fake_client(list_json, _multi_metric_run(latency, instruction)) as client:
        await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=IDS,
            test_set_id="TS1",
            noisy_persona_id="PN",
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
        )
    datasets = [c.kwargs["dataset_id"] for c in writer.start_run.await_args_list]
    assert datasets == ["s2s-multiturn-noisy-v1", "s2s-multiturn-v1"]
    # Both runs carry latency, but the noise dataset excludes it: only the clean
    # run writes V2V rows, and neither warns about the omission.
    written = [c.args[0] for c in writer.record_results.await_args_list]
    assert [{r.metric_type for r in rows} for rows in written] == [
        {Metric.INSTRUCTION_FOLLOWING},
        {Metric.V2V, Metric.INSTRUCTION_FOLLOWING},
    ]
    # Provenance rides along so a row says which persona produced it.
    personas = [c.kwargs["persona_id"] for c in writer.start_run.await_args_list]
    assert personas == ["PN", "PCLEAN"]


@pytest.mark.asyncio
async def test_noisy_caller_never_becomes_a_sample_candidate() -> None:
    """Embargoed audio must not reach the public samples card."""
    from coval_bench.s2s.samples import SampleRun

    writer = _stub_writer()
    list_json = _list_json(
        {"run_id": "RNOISY", "create_time": _iso(timedelta(hours=1)), "persona_id": "PN"},
        {"run_id": "RCLEAN", "create_time": _iso(timedelta(hours=1)), "persona_id": "PCLEAN"},
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    sampled: list[SampleRun] = []
    async with _fake_client(list_json, _run_json(values)) as client:
        await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            test_set_id="TS1",
            noisy_persona_id="PN",
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
        )
    assert [r.coval_run_id for r in sampled] == ["RCLEAN"]


@pytest.mark.asyncio
async def test_fetch_and_write_rejects_noisy_persona_without_a_test_set() -> None:
    # Without a test set the persona split can never fire, so fail loudly.
    settings = Settings(
        runner_sha="test",
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_noisy_persona_id="PN",
    )
    with pytest.raises(RuntimeError, match="requires coval_s2s_test_set_id"):
        await fetch_v2v.fetch_and_write_v2v(settings)


@pytest.mark.asyncio
async def test_fetch_and_write_rejects_a_blank_happypath_test_set() -> None:
    # Blank would skip its agents with only a warning, indistinguishable from unset.
    settings = Settings(
        runner_sha="test",
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_happypath_test_set_id="   ",
    )
    with pytest.raises(RuntimeError, match="coval_s2s_happypath_test_set_id must not be blank"):
        await fetch_v2v.fetch_and_write_v2v(settings)


@pytest.mark.asyncio
async def test_fetch_and_write_rejects_a_metric_with_no_row_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a configured metric whose mapper is missing.
    mappers = dict(fetch_v2v._VALUE_MAPPERS)
    del mappers[Metric.INTERRUPTION_RATE]
    monkeypatch.setattr(fetch_v2v, "_VALUE_MAPPERS", mappers)
    settings = Settings(
        runner_sha="test",
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_test_set_id="TS1",
        coval_s2s_instruction_metric_id="IID",
        coval_s2s_interruption_metric_id="RID",
    )
    with pytest.raises(RuntimeError, match="no _VALUE_MAPPERS entry"):
        await fetch_v2v.fetch_and_write_v2v(settings)


def test_interruption_value_maps() -> None:
    assert fetch_v2v._interruption_value(1.1666666666666632) == (1.17, ResultStatus.SUCCESS)
    assert fetch_v2v._interruption_value(0) == (0.0, ResultStatus.SUCCESS)
    assert fetch_v2v._interruption_value(None) == (None, ResultStatus.FAILED)
    assert fetch_v2v._interruption_value("1.5") == (None, ResultStatus.FAILED)
    assert fetch_v2v._interruption_value(True) == (None, ResultStatus.FAILED)


@pytest.mark.asyncio
async def test_ingest_run_writes_interruption_rows() -> None:
    writer = _stub_writer()
    latency = [{"simulation_output_id": f"s{i}", "value": 0.5} for i in range(2)]
    instruction = [{"simulation_output_id": f"s{i}", "value": "YES"} for i in range(2)]
    interruption = [
        {"simulation_output_id": "s0", "value": 1.25},
        {"simulation_output_id": "s1", "value": None},
    ]
    run_json = {
        "run": {
            "error_status": "SUCCESS",
            "results": {
                "metrics": {
                    "MID": {"values": latency},
                    "IID": {"values": instruction},
                    "RID": {"values": interruption},
                }
            },
        }
    }
    async with _fake_client({}, run_json) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=ALL_IDS,
            condition=condition_for(DATASET_ID_MULTITURN),
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    irr = [r for r in rows if r.metric_type == Metric.INTERRUPTION_RATE]
    assert [(r.audio_filename, r.metric_value, r.status) for r in irr] == [
        ("R1/s0", 1.25, ResultStatus.SUCCESS),
        ("R1/s1", None, ResultStatus.FAILED),
    ]
    assert all(r.metric_units == "per_minute" for r in irr)


@pytest.mark.asyncio
async def test_ingest_run_writes_instruction_rows() -> None:
    writer = _stub_writer()
    latency = [{"simulation_output_id": f"s{i}", "value": 0.5} for i in range(3)]
    instruction = [
        {"simulation_output_id": "s0", "value": "YES"},
        {"simulation_output_id": "s1", "value": "NO"},
        {"simulation_output_id": "s2", "value": "UNKNOWN"},
    ]
    run_json = {
        "run": {
            "error_status": "SUCCESS",
            "results": {"metrics": {"MID": {"values": latency}, "IID": {"values": instruction}}},
        }
    }
    async with _fake_client({}, run_json) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            runner_sha="test",
            period_seconds=10_800,
        )
    # Run status reflects latency (all numeric) -> SUCCEEDED.
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    latency_rows = [r for r in rows if r.metric_type == Metric.V2V]
    instr_rows = [r for r in rows if r.metric_type == Metric.INSTRUCTION_FOLLOWING]
    assert len(latency_rows) == 3
    assert len(instr_rows) == 2  # UNKNOWN (s2) excluded from the pool
    by_sim = {r.audio_filename: (r.metric_value, r.status) for r in instr_rows}
    assert by_sim["R1/s0"] == (100.0, ResultStatus.SUCCESS)
    assert by_sim["R1/s1"] == (0.0, ResultStatus.SUCCESS)
    assert "R1/s2" not in by_sim  # UNKNOWN produced no row


def _multi_metric_run(
    latency: list[dict[str, Any]], instruction: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "run": {
            "error_status": "SUCCESS",
            "results": {"metrics": {"MID": {"values": latency}, "IID": {"values": instruction}}},
        }
    }


@pytest.mark.asyncio
async def test_ingest_run_id_mismatch_keeps_latency() -> None:
    writer = _stub_writer()
    latency = [
        {"simulation_output_id": "s0", "value": 0.5},
        {"simulation_output_id": "s1", "value": 0.5},
    ]
    instruction = [
        {"simulation_output_id": "s0", "value": "YES"},
        {"simulation_output_id": "s2", "value": "YES"},  # s2 not in latency -> mismatch
    ]
    async with _fake_client({}, _multi_metric_run(latency, instruction)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED  # latency intact
    rows = writer.record_results.await_args.args[0]
    assert all(r.metric_type == Metric.V2V for r in rows)  # instruction skipped on mismatch


@pytest.mark.asyncio
async def test_ingest_run_invalid_verdict_discards_instruction() -> None:
    writer = _stub_writer()
    latency = [{"simulation_output_id": "s0", "value": 0.5}]
    instruction = [{"simulation_output_id": "s0", "value": "GARBAGE"}]
    async with _fake_client({}, _multi_metric_run(latency, instruction)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED  # latency kept
    rows = writer.record_results.await_args.args[0]
    assert all(r.metric_type == Metric.V2V for r in rows)  # instruction discarded


@pytest.mark.asyncio
async def test_fetch_and_write_requires_id_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    # instruction id set but test-set id missing -> startup failure (must be paired).
    settings = Settings(
        runner_sha="test",
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_instruction_metric_id="IID",
    )
    with pytest.raises(RuntimeError, match="set together"):
        await fetch_v2v.fetch_and_write_v2v(settings)


@pytest.mark.asyncio
async def test_ingest_run_backfill_instruction_only() -> None:
    # Latency already ingested (want_latency=False): backfill only instruction.
    writer = _stub_writer()
    latency = [{"simulation_output_id": "s0", "value": 0.5}]
    instruction = [{"simulation_output_id": "s0", "value": "YES"}]
    async with _fake_client({}, _multi_metric_run(latency, instruction)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            pending=frozenset({Metric.INSTRUCTION_FOLLOWING}),
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    assert len(rows) == 1
    assert all(r.metric_type == Metric.INSTRUCTION_FOLLOWING for r in rows)  # no latency rewrite


@pytest.mark.asyncio
async def test_ingest_run_backfill_instruction_absent_is_noop() -> None:
    # Backfill wanted but the instruction metric isn't on the run yet -> retryable no-op.
    writer = _stub_writer()
    latency = [{"simulation_output_id": "s0", "value": 0.5}]
    async with _fake_client({}, _run_json(latency)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            pending=frozenset({Metric.INSTRUCTION_FOLLOWING}),
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is None  # nothing to write -> no run row, stays retryable
    writer.start_run.assert_not_awaited()
    writer.record_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_run_latency_absent_writes_instruction() -> None:
    # The noise condition requires instruction, not latency, so a run with no V2V
    # is normal there and its instruction rows must still land.
    writer = _stub_writer()
    instruction = [{"simulation_output_id": "s0", "value": "YES"}]
    async with _fake_client({}, _run_json(instruction, metric_id="IID")) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN_NOISY),
            dataset_id=DATASET_ID_MULTITURN_NOISY,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    assert len(rows) == 1
    assert all(r.metric_type == Metric.INSTRUCTION_FOLLOWING for r in rows)


@pytest.mark.asyncio
async def test_ingest_run_latency_required_on_the_standard_caller() -> None:
    # Same payload under the standard condition: latency is a must, so this is a
    # fault and nothing is written rather than instruction publishing on its own.
    writer = _stub_writer()
    instruction = [{"simulation_output_id": "s0", "value": "YES"}]
    async with _fake_client({}, _run_json(instruction, metric_id="IID")) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN),
            dataset_id=DATASET_ID_MULTITURN,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is None
    writer.start_run.assert_not_awaited()
    writer.record_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_run_rejects_duplicate_ids_in_the_anchor() -> None:
    # Noise condition, so instruction is the anchor and nothing compares it against
    # another metric: this guard is the only thing stopping double-counted rows.
    writer = _stub_writer()
    instruction = [
        {"simulation_output_id": "s0", "value": "YES"},
        {"simulation_output_id": "s0", "value": "NO"},
    ]
    async with _fake_client({}, _run_json(instruction, metric_id="IID")) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN_NOISY),
            dataset_id=DATASET_ID_MULTITURN_NOISY,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is None
    writer.start_run.assert_not_awaited()
    writer.record_results.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "instruction",
    [[], [{"simulation_output_id": "s0", "value": "UNKNOWN"}]],
    ids=["empty", "all-unknown"],
)
async def test_ingest_run_latency_absent_instruction_without_rows_is_noop(
    instruction: list[dict[str, Any]],
) -> None:
    # Noise condition: instruction is the anchor, so values mapping to no rows
    # must not leave an empty run row behind.
    writer = _stub_writer()
    async with _fake_client({}, _run_json(instruction, metric_id="IID")) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN_NOISY),
            dataset_id=DATASET_ID_MULTITURN_NOISY,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is None
    writer.start_run.assert_not_awaited()
    writer.record_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_ingested_instruction_only_run_is_fresh() -> None:
    # Noise requires instruction and never carries latency, so a persisted
    # instruction run keeps the provider fresh between fetches.
    writer = _stub_writer()

    async def metric_ingested(*, metric_type: str, **_kwargs: Any) -> bool:
        return metric_type == Metric.INSTRUCTION_FOLLOWING

    writer.coval_metric_ingested = AsyncMock(side_effect=metric_ingested)
    list_json = _list_json(
        {"run_id": "R1", "create_time": _iso(timedelta(hours=1)), "persona_id": "PN"}
    )
    instruction = [{"simulation_output_id": "s0", "value": "YES"}]
    async with _fake_client(list_json, _run_json(instruction, metric_id="IID")) as client:
        status, ingested = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=IDS,
            test_set_id="TS1",
            noisy_persona_id="PN",
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
        )

    assert (status, ingested) == (RunStatus.SUCCEEDED, 0)
    writer.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_skipped_when_the_required_metric_is_unconfigured() -> None:
    # Noise requires instruction; without its id the run cannot be judged, so it is
    # skipped and must not count as fresh data or a sample candidate.
    from coval_bench.s2s.samples import SampleRun

    writer = _stub_writer()
    list_json = _list_json(
        {"run_id": "RNOISY", "create_time": _iso(timedelta(hours=1)), "persona_id": "PN"}
    )
    sampled: list[SampleRun] = []
    async with _fake_client(list_json, _run_json([{"simulation_output_id": "s1"}])) as client:
        status, ingested = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            test_set_id="TS1",
            noisy_persona_id="PN",
            runner_sha="test",
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
        )
    assert ingested == 0
    assert sampled == []
    writer.start_run.assert_not_awaited()
    assert status is not RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_ingest_run_no_metrics_present_is_noop() -> None:
    # Neither metric on the run -> no run row, stays retryable.
    writer = _stub_writer()
    async with _fake_client({}, _run_json([], metric_id="OTHER")) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=IDS,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is None
    writer.start_run.assert_not_awaited()
    writer.record_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_run_without_instruction_metric_id() -> None:
    # No instruction metric id -> only latency rows, no crash.
    writer = _stub_writer()
    latency = [{"simulation_output_id": "s0", "value": 0.5}]
    async with _fake_client({}, _run_json(latency)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None, error_status=None),
            metric_ids=LATENCY_IDS,
            runner_sha="test",
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    assert all(r.metric_type == Metric.V2V for r in rows)


@pytest.mark.asyncio
async def test_recent_completed_runs_filters_by_test_set() -> None:
    captured: list[httpx.Request] = []
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})
    async with _fake_client(list_json, {}, captured) as client:
        await fetch_v2v.recent_completed_runs(
            client, "a1", period_seconds=10_800, test_set_id="TS1"
        )
    filter_expr = captured[0].url.params["filter"]
    assert 'test_set_id="TS1"' in filter_expr
    assert 'agent_id="a1"' in filter_expr


def test_log_run_partial_emits_run_partial_event() -> None:
    # The infra partial-alert metric greps for this exact event string.
    with capture_logs() as logs:
        log_run_partial("s2s fetch has no fresh data from: openai")
    assert [entry["event"] for entry in logs] == ["RUN_PARTIAL"]


def test_log_run_failed_emits_run_failed_event() -> None:
    # Unchanged contract shared with the STT/TTS orchestrator's failure metric.
    with capture_logs() as logs:
        log_run_failed("s2s fetch failed for all providers")
    assert [entry["event"] for entry in logs] == ["RUN_FAILED"]


def _run_fetch_cli(
    monkeypatch: pytest.MonkeyPatch, statuses: dict[str, RunStatus]
) -> tuple[int, list[str]]:
    async def fake_fetch(_settings: Settings) -> dict[str, RunStatus]:
        return statuses

    settings = Settings.model_construct(log_level="INFO")
    monkeypatch.setattr(fetch_v2v, "fetch_and_write_v2v", fake_fetch)
    monkeypatch.setattr(fetch_v2v, "get_settings", lambda: settings)
    monkeypatch.setattr("coval_bench.logging.configure_logging", lambda level: None)
    with capture_logs() as logs:
        result = CliRunner().invoke(fetch_v2v.fetch_s2s, [])
    return result.exit_code, [str(entry.get("event")) for entry in logs]


def test_cli_mixed_alerts_partial_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    code, events = _run_fetch_cli(
        monkeypatch, {"openai": RunStatus.SUCCEEDED, "google": RunStatus.FAILED}
    )
    assert code == 0
    assert "RUN_PARTIAL" in events
    assert "RUN_FAILED" not in events


def test_cli_all_failed_alerts_failed_exit_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    code, events = _run_fetch_cli(
        monkeypatch, {"openai": RunStatus.FAILED, "google": RunStatus.FAILED}
    )
    assert code != 0
    assert "RUN_FAILED" in events
    assert "RUN_PARTIAL" not in events


def test_cli_no_providers_alerts_failed_exit_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    code, events = _run_fetch_cli(monkeypatch, {})
    assert code != 0
    assert "RUN_FAILED" in events
    assert "RUN_PARTIAL" not in events
