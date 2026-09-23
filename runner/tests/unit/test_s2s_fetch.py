# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the S2S V2V fetch job."""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from click.testing import CliRunner
from structlog.testing import capture_logs

from coval_bench.config import ScenarioCovalIds, Settings
from coval_bench.db.models import MetricExecutor, ResultStatus, Run, RunStatus
from coval_bench.logging import log_run_failed, log_run_partial, log_run_unmapped_persona
from coval_bench.registries import Benchmark, Metric
from coval_bench.registries.models import RegisteredModel
from coval_bench.runner.capture import (
    ImportRunClaim,
    RunSeal,
    build_capture_identity,
    identity_digest,
)
from coval_bench.s2s import fetch_v2v
from coval_bench.s2s.conditions import (
    DATASET_ID_BANK,
    DATASET_ID_BANK_LOW,
    DATASET_ID_DENTAL,
    DATASET_ID_LLM_BANK,
    DATASET_ID_MULTITURN,
    DATASET_ID_MULTITURN_NOISY,
    DEFAULT_CONDITION,
    FAMILY_BANK,
    FAMILY_DENTAL,
    FAMILY_HAPPYPATH,
    FAMILY_LLM_BANK,
    FAMILY_MULTITURN,
    SCENARIO_SLUGS,
    Condition,
    DatasetMetrics,
    condition_for,
)
from coval_bench.s2s.fetch_v2v import AgentSpec, CovalRun

IDS = {Metric.V2V: "MID", Metric.INSTRUCTION_FOLLOWING: "IID"}
LATENCY_IDS = {Metric.V2V: "MID"}
ALL_IDS = {**IDS, Metric.INTERRUPTION_RATE: "RID"}

SPEC = AgentSpec(agent_id="a1", provider="openai", model="gpt-realtime")


def _every_s2s_row() -> list[RegisteredModel]:
    """One unpublished registry row per S2S spec, as production has."""
    agent_fields: dict[str, Any] = {
        name: "x"
        for name in Settings.model_fields
        if name.startswith("coval_s2s_") and name.endswith("_agent_id")
    }
    agent_fields["coval_s2s_scenarios"] = {
        slug: ScenarioCovalIds(
            test_set_id="x", agents={model: "x" for _provider, model in fetch_v2v.SCENARIO_MODELS}
        )
        for slug in SCENARIO_SLUGS
    }
    return [
        RegisteredModel(
            benchmark=Benchmark.S2S,
            provider=spec.provider,
            model=spec.model,
            collected=True,
            published=False,
        )
        for spec in fetch_v2v.s2s_specs(Settings.model_construct(**agent_fields))
    ]


@pytest.fixture(autouse=True)
def _registry_has_every_s2s_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch_v2v, "fetch_models", AsyncMock(return_value=_every_s2s_row()))


PHONELY = RegisteredModel(
    benchmark=Benchmark.LLM,
    provider="phonely",
    model="phonely-agent",
    collected=True,
    published=False,
)
LLM_SPEC = AgentSpec(
    agent_id="a1",
    provider="phonely",
    model="phonely-agent",
    test_set_id_attr="coval_s2s_bank_test_set_id",
    family=FAMILY_LLM_BANK,
    publish_samples=False,
    benchmark=Benchmark.LLM,
    instruction_metric_id_attr="coval_s2s_bank_instruction_metric_id",
)


def _iso(age: timedelta) -> str:
    return (datetime.now(tz=UTC) - age).isoformat().replace("+00:00", "Z")


def _list_json(*runs: dict[str, Any]) -> dict[str, Any]:
    return {"runs": list(runs)}


def _run_json(
    values: list[dict[str, Any]],
    metric_id: str = "MID",
    error_status: str | None = "SUCCESS",
    output_ids: list[str] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {"metrics": {metric_id: {"values": values}}}
    if output_ids is not None:
        results["output_ids"] = output_ids
    run: dict[str, Any] = {"results": results}
    if error_status:
        run["error_status"] = error_status
    return {"run": run}


def _fake_client(
    list_json: dict[str, Any] | list[dict[str, Any]],
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
            if isinstance(list_json, list):
                assert captured is not None
                page = len([r for r in captured if r.url.path.endswith("/runs")]) - 1
                payload = dict(list_json[min(page, len(list_json) - 1)])
                if page + 1 < len(list_json):
                    payload.update({"next_page_" + "token": "next-page"})
                return httpx.Response(200, json=payload)
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
            dataset_id="s2s-v1",
            dataset_sha256="x",
            status=RunStatus.RUNNING,
        )
    )
    writer.coval_metric_ingested = AsyncMock(return_value=False)
    writer.get_run = AsyncMock(return_value=writer.start_run.return_value)
    writer.reserve_run_id = AsyncMock(return_value=1)
    writer.ensure_capture_run = AsyncMock(return_value=writer.start_run.return_value)
    writer.conversation_ttft = AsyncMock(return_value={})
    writer.record_results = AsyncMock()
    writer.finish_run = AsyncMock()
    writer.refresh_bucket = AsyncMock()
    writer.refresh_metric_values_bucket = AsyncMock()
    writer.refresh_stats_matviews = AsyncMock()
    writer.refresh_dashboard_summaries = AsyncMock(return_value="published")
    return writer


async def _fetch(client: httpx.AsyncClient, writer: MagicMock) -> tuple[RunStatus, int]:
    return await fetch_v2v._fetch_one_provider(
        client,
        writer,
        spec=SPEC,
        agent_id="a1",
        metric_ids=LATENCY_IDS,
        period_seconds=10_800,
        stale_grace_seconds=5_400,
    )


@pytest.mark.asyncio
async def test_recent_completed_runs_window_and_page_overrides() -> None:
    captured: list[httpx.Request] = []
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(days=20))})
    async with _fake_client(list_json, {}, captured) as client:
        runs = await fetch_v2v.recent_completed_runs(
            client, "a1", period_seconds=10_800, window_seconds=30 * 86_400, page_size=100
        )

    assert [r.run_id for r in runs] == ["R1"]
    assert captured[0].url.params["page_size"] == "100"


@pytest.mark.asyncio
async def test_recent_completed_runs_follows_targeted_pagination() -> None:
    captured: list[httpx.Request] = []
    pages = [
        _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))}),
        _list_json({"run_id": "R2", "create_time": _iso(timedelta(hours=2))}),
    ]
    async with _fake_client(pages, {}, captured) as client:
        runs = await fetch_v2v.recent_completed_runs(
            client,
            "a1",
            period_seconds=10_800,
            window_seconds=30 * 86_400,
            page_size=1,
            requested_run_ids=frozenset({"R2"}),
        )

    assert {r.run_id for r in runs} == {"R1", "R2"}
    assert len(captured) == 2
    assert captured[1].url.params["page_token"] == "-".join(("next", "page"))


@pytest.mark.asyncio
async def test_recent_completed_runs_pages_until_the_window_is_left() -> None:
    """Five personas a day fill a ten-run page in a two-day window; the scan must
    keep paging while every run is still inside the window, and stop once a
    page reaches past it."""
    captured: list[httpx.Request] = []
    pages = [
        _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))}),
        _list_json({"run_id": "R2", "create_time": _iso(timedelta(hours=20))}),
        _list_json(
            {"run_id": "R3", "create_time": _iso(timedelta(hours=40))},
            {"run_id": "R4", "create_time": _iso(timedelta(days=3))},
        ),
        _list_json({"run_id": "R5", "create_time": _iso(timedelta(days=4))}),
    ]
    async with _fake_client(pages, {}, captured) as client:
        runs = await fetch_v2v.recent_completed_runs(
            client, "a1", period_seconds=86_400, page_size=1
        )

    assert [r.run_id for r in runs] == ["R1", "R2", "R3"]
    assert len(captured) == 3


@pytest.mark.asyncio
async def test_backfill_ingests_only_named_runs() -> None:
    writer = _stub_writer()
    list_json = _list_json(
        {"run_id": "R2", "create_time": _iso(timedelta(hours=1))},
        {"run_id": "R1", "create_time": _iso(timedelta(hours=4))},
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    async with _fake_client(list_json, _run_json(values)) as client:
        status, ingested = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            only_run_ids=frozenset({"R1"}),
        )

    assert (status, ingested) == (RunStatus.SUCCEEDED, 1)
    written = [c.args[0][0].audio_filename for c in writer.record_results.await_args_list]
    assert written == ["R1/s1"]


@pytest.mark.asyncio
async def test_backfill_reports_matches_and_ignores_staleness() -> None:
    writer = _stub_writer()
    matched: set[str] = set()
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(days=20))})
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    async with _fake_client(list_json, _run_json(values)) as client:
        status, ingested = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            only_run_ids=frozenset({"R1", "R9"}),
            window_seconds=30 * 86_400,
            matched_run_ids=matched,
        )

    assert (status, ingested) == (RunStatus.SUCCEEDED, 1)
    assert matched == {"R1"}


@pytest.mark.asyncio
async def test_backfill_matches_errored_run() -> None:
    # The EXECUTION_FAILURE stamp no longer gates ingestion: a backfilled run
    # lands its healthy clips (plus FAILED rows for the rest) and is matched.
    writer = _stub_writer()
    matched: set[str] = set()
    list_json = _list_json(
        {"run_id": "R1", "create_time": _iso(timedelta(hours=1)), "error_status": "FAILED"}
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    errored = _run_json(values, error_status="EXECUTION_FAILURE", output_ids=["s1", "s2"])
    async with _fake_client(list_json, errored) as client:
        status, ingested = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            only_run_ids=frozenset({"R1"}),
            matched_run_ids=matched,
        )

    assert (status, ingested) == (RunStatus.PARTIAL, 1)
    assert matched == {"R1"}


@pytest.mark.asyncio
async def test_backfill_does_not_match_failed_ingest() -> None:
    writer = _stub_writer()
    matched: set[str] = set()
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})
    values = [{"simulation_output_id": "s1", "value": "not-a-number"}]
    async with _fake_client(list_json, _run_json(values)) as client:
        status, ingested = await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            only_run_ids=frozenset({"R1"}),
            matched_run_ids=matched,
        )

    assert (status, ingested) == (RunStatus.FAILED, 1)
    assert matched == set()


@pytest.mark.asyncio
async def test_backfill_publishes_no_samples() -> None:
    from coval_bench.s2s.samples import SampleRun

    writer = _stub_writer()
    sampled: list[SampleRun] = []
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    async with _fake_client(list_json, _run_json(values)) as client:
        await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=SPEC,
            agent_id="a1",
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
            only_run_ids=frozenset({"R1"}),
        )

    assert sampled == []


def test_expected_sample_models_excludes_non_publishing_agents() -> None:
    settings = Settings.model_construct(
        coval_s2s_openai_agent_id="a1",
        coval_s2s_gray_agent_id="a2",
        coval_s2s_red_agent_id="a3",
    )
    expected = fetch_v2v._expected_sample_models(settings, DATASET_ID_DENTAL)

    assert ("openai", "gpt-realtime") in expected
    assert ("colors", "gray") not in expected
    assert ("colors", "red") not in expected
    assert ("phonely", "phonely-agent") not in expected


def test_expected_sample_models_are_scoped_to_the_dataset_partition() -> None:
    """An agent counts only on its own dataset: a model absent from another set is
    not a missing provider there."""
    settings = Settings.model_construct(
        coval_s2s_openai_agent_id="a1",
        coval_s2s_scenarios={
            "bank": ScenarioCovalIds(
                test_set_id="TSB",
                agents={
                    "gpt-realtime": "b1",
                    "gpt-live-1": "b2",
                    "gemini-live": "b3",
                    "grok-voice-think-fast-2.0": "b4",
                    "stepaudio-3-realtime-preview": "b5",
                },
            )
        },
    )

    assert fetch_v2v._expected_sample_models(settings, DATASET_ID_DENTAL) == {
        ("openai", "gpt-realtime")
    }
    assert fetch_v2v._expected_sample_models(settings, DATASET_ID_BANK) == {
        ("openai", "gpt-realtime"),
        ("openai", "gpt-live-1"),
        ("google", "gemini-live"),
        ("xai", "grok-voice-think-fast-2.0"),
        ("stepfun", "stepaudio-3-realtime-preview"),
    }


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
            coval_run=CovalRun(run_id="R1", create_time=created),
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    assert writer.start_run.await_args.kwargs["scheduled_at"] == datetime(2026, 7, 7, 0, tzinfo=UTC)
    writer.record_results.assert_awaited_once()
    writer.refresh_bucket.assert_awaited_once()
    writer.refresh_metric_values_bucket.assert_awaited_once_with(1)
    assert writer.finish_run.await_args.kwargs["status"] is RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_ingest_run_partial_and_failed() -> None:
    writer = _stub_writer()
    mixed: list[dict[str, Any]] = [
        {"simulation_output_id": "s1", "value": 0.5},
        {"simulation_output_id": "s2", "value": None},
    ]
    run = CovalRun(run_id="R1", create_time=None)
    async with _fake_client({}, _run_json(mixed)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=run,
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.PARTIAL
    writer.refresh_bucket.assert_awaited_once()
    writer.refresh_metric_values_bucket.assert_awaited_once_with(1)

    writer = _stub_writer()
    all_null: list[dict[str, Any]] = [{"simulation_output_id": "s1", "value": None}]
    async with _fake_client({}, _run_json(all_null)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=run,
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.FAILED
    writer.refresh_bucket.assert_not_awaited()
    writer.refresh_metric_values_bucket.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_refresh", ["refresh_bucket", "refresh_metric_values_bucket"])
async def test_ingest_run_rollup_refresh_failure_does_not_change_status(
    failing_refresh: str,
) -> None:
    """Either best-effort rollup failure leaves the completed run succeeded."""
    writer = _stub_writer()
    getattr(writer, failing_refresh).side_effect = RuntimeError("db down")
    values = [{"simulation_output_id": "s1", "value": 0.5}]

    async with _fake_client({}, _run_json(values)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
        )

    assert status is RunStatus.SUCCEEDED
    writer.finish_run.assert_awaited_once_with(1, status=RunStatus.SUCCEEDED)
    writer.refresh_bucket.assert_awaited_once()
    writer.refresh_metric_values_bucket.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_ingest_run_skips_before_any_write() -> None:
    # Metric absent: skipped, no run row created.
    writer = _stub_writer()
    run = CovalRun(run_id="R1", create_time=None)
    async with _fake_client({}, _run_json([], metric_id="OTHER")) as client:
        assert (
            await fetch_v2v._ingest_run(
                client,
                writer,
                spec=SPEC,
                coval_run=run,
                metric_ids=LATENCY_IDS,
                period_seconds=10_800,
            )
            is None
        )
    writer.start_run.assert_not_awaited()

    # Anchor present but valueless (a fully-wrecked run): still skipped.
    writer = _stub_writer()
    async with _fake_client({}, _run_json([], output_ids=["s1", "s2"])) as client:
        assert (
            await fetch_v2v._ingest_run(
                client,
                writer,
                spec=SPEC,
                coval_run=run,
                metric_ids=LATENCY_IDS,
                period_seconds=10_800,
            )
            is None
        )
    writer.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_run_ignores_error_status() -> None:
    # One failed conversation stamps the whole run EXECUTION_FAILURE; the
    # surviving clips must still land.
    writer = _stub_writer()
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    async with _fake_client({}, _run_json(values, error_status="EXECUTION_FAILURE")) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    writer.record_results.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_run_failed_conversations_become_failed_rows() -> None:
    # s3 is in output_ids but has no anchor value: it failed on Coval's side
    # and must land as a FAILED row, making the run PARTIAL.
    writer = _stub_writer()
    values = [
        {"simulation_output_id": "s1", "value": 0.5},
        {"simulation_output_id": "s2", "value": 0.6},
    ]
    fixture = _run_json(values, output_ids=["s1", "s2", "s3"])
    async with _fake_client({}, fixture) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.PARTIAL
    rows = writer.record_results.await_args.args[0]
    assert [(r.audio_filename, r.status) for r in rows] == [
        ("R1/s1", ResultStatus.SUCCESS),
        ("R1/s2", ResultStatus.SUCCESS),
        ("R1/s3", ResultStatus.FAILED),
    ]
    assert rows[-1].metric_value is None


@pytest.mark.asyncio
async def test_ingest_run_anchor_without_id_synthesizes_no_failures() -> None:
    # One anchor value has no simulation_output_id (kept under the index
    # fallback), so its output_id can't be matched: no output_id may be treated
    # as uncovered, or a measured conversation would be stamped FAILED.
    writer = _stub_writer()
    values: list[dict[str, Any]] = [
        {"simulation_output_id": "s1", "value": 0.5},
        {"value": 0.6},  # this is s2, but the id is missing
    ]
    fixture = _run_json(values, output_ids=["s1", "s2"])
    async with _fake_client({}, fixture) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=LATENCY_IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    assert [(r.audio_filename, r.status) for r in rows] == [
        ("R1/s1", ResultStatus.SUCCESS),
        ("R1/1", ResultStatus.SUCCESS),
    ]


@pytest.mark.asyncio
async def test_fetch_one_provider_ingests_every_new_run() -> None:
    writer = _stub_writer()
    writer.start_run = AsyncMock(
        side_effect=[
            Run(
                id=i,
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
async def test_pending_metrics_include_local_metrics_and_scope_dedupe_to_benchmark() -> None:
    writer = _stub_writer()
    condition = DatasetMetrics(
        benchmark=Benchmark.LLM,
        required=Metric.INSTRUCTION_FOLLOWING,
        local=frozenset({Metric.TTFT}),
    )

    pending = await fetch_v2v._pending_metrics(
        writer,
        benchmark=Benchmark.LLM,
        provider="phonely",
        coval_run_id="R1",
        condition=condition,
        metric_ids={Metric.INSTRUCTION_FOLLOWING: "IID"},
    )

    assert pending == frozenset({Metric.INSTRUCTION_FOLLOWING, Metric.TTFT})
    assert {call.kwargs["benchmark"] for call in writer.coval_metric_ingested.await_args_list} == {
        Benchmark.LLM
    }
    assert {
        call.kwargs["metric_type"] for call in writer.coval_metric_ingested.await_args_list
    } == {
        Metric.INSTRUCTION_FOLLOWING,
        Metric.TTFT,
    }


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

    async def ingested(
        *, benchmark: Benchmark, provider: str, coval_run_id: str, metric_type: Metric
    ) -> bool:
        assert benchmark is Benchmark.S2S
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
async def test_fetch_one_provider_ingests_errored_run() -> None:
    # An EXECUTION_FAILURE stamp (one failed conversation) no longer drops the
    # run: its healthy clips ingest and count as freshness.
    writer = _stub_writer()
    writer.start_run = AsyncMock(
        side_effect=[
            Run(
                id=i,
                dataset_id="s2s-v1",
                dataset_sha256="x",
                status=RunStatus.RUNNING,
            )
            for i in (1, 2)
        ]
    )
    list_json = _list_json(
        {
            "run_id": "R2",
            "create_time": _iso(timedelta(hours=1)),
            "error_status": "EXECUTION_FAILURE",
        },
        {"run_id": "R1", "create_time": _iso(timedelta(hours=4))},
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    errored = _run_json(values, error_status="EXECUTION_FAILURE", output_ids=["s1", "s2"])
    async with _fake_client(list_json, {"R2": errored, "R1": _run_json(values)}, None) as client:
        status, ingested = await _fetch(client, writer)

    assert (status, ingested) == (RunStatus.PARTIAL, 2)
    written = [c.args[0][0].audio_filename for c in writer.record_results.await_args_list]
    assert written == ["R2/s1", "R1/s1"]
    partial = writer.finish_run.await_args_list[0].kwargs["status"]
    assert partial is RunStatus.PARTIAL


@pytest.mark.asyncio
async def test_fetch_and_write_v2v_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # gemini agent id left unset (None) -> that provider is skipped.
    monkeypatch.delenv("COVAL_S2S_GEMINI_AGENT_ID", raising=False)
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_dental_test_set_id="TSD",
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
    assert statuses == {"s2s-dental:openai:gpt-realtime": RunStatus.SUCCEEDED}
    writer.refresh_stats_matviews.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_and_write_skips_an_arm_with_no_registry_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_scenarios={
            "bank": ScenarioCovalIds(
                test_set_id="TSB",
                agents={"stepaudio-3-realtime-preview": "b5"},
                personas={"clean": "PID"},
            )
        },
        coval_s2s_bank_test_set_id="TSB",
        coval_s2s_bank_instruction_metric_id="IID",
        coval_s2s_bank_persona_id="PID",
    )
    client = _fake_client({}, {})
    writer = _stub_writer()

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    fetch_one = AsyncMock(return_value=(RunStatus.SUCCEEDED, 0))
    without_stepfun = [m for m in _every_s2s_row() if m.provider != "stepfun"]
    monkeypatch.setattr(fetch_v2v, "fetch_models", AsyncMock(return_value=without_stepfun))
    monkeypatch.setattr(fetch_v2v, "_client", lambda _settings: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)
    monkeypatch.setattr(fetch_v2v, "_fetch_one_provider", fetch_one)

    statuses = await fetch_v2v.fetch_and_write_v2v(settings)

    assert statuses == {}
    fetch_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_and_write_filters_agents_and_allows_llm_without_v2v(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        coval_s2s_openai_agent_id="s2s-agent",
        coval_s2s_bank_test_set_id="TSB",
        coval_s2s_bank_instruction_metric_id="IID",
        coval_s2s_bank_persona_id="PID",
    )
    client = _fake_client({}, {})
    writer = _stub_writer()

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    fetch_one = AsyncMock(return_value=(RunStatus.SUCCEEDED, 0))
    paused = RegisteredModel(
        benchmark=Benchmark.LLM,
        provider="paused",
        model="paused-agent",
        collected=False,
        published=False,
    )
    monkeypatch.setattr(fetch_v2v, "s2s_specs", lambda _s: (SPEC,))
    monkeypatch.setattr(fetch_v2v, "fetch_models", AsyncMock(return_value=[PHONELY, paused]))
    monkeypatch.setattr(fetch_v2v, "_client", lambda _settings: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)
    monkeypatch.setattr(fetch_v2v, "_fetch_one_provider", fetch_one)

    statuses = await fetch_v2v.fetch_and_write_v2v(
        settings, benchmark=Benchmark.LLM, llm_agent_ids={"phonely": "llm-agent"}
    )

    assert statuses == {"llm-bank:phonely:phonely-agent": RunStatus.SUCCEEDED}
    fetch_one.assert_awaited_once()
    assert fetch_one.await_args is not None
    assert fetch_one.await_args.kwargs["spec"].agent_id == "llm-agent"
    assert fetch_one.await_args.kwargs["agent_id"] == "llm-agent"
    assert fetch_one.await_args.kwargs["metric_ids"] == {Metric.INSTRUCTION_FOLLOWING: "IID"}


def test_phonely_spec_is_the_llm_bank_text_agent() -> None:
    (spec,) = fetch_v2v.llm_specs(
        [PHONELY, PHONELY.model_copy(update={"collected": False})], {"phonely": "A1"}
    )
    assert spec == AgentSpec(
        agent_id="A1",
        provider="phonely",
        model="phonely-agent",
        test_set_id_attr="coval_s2s_bank_test_set_id",
        family=FAMILY_LLM_BANK,
        publish_samples=False,
        benchmark=Benchmark.LLM,
        instruction_metric_id_attr="coval_s2s_bank_instruction_metric_id",
    )
    assert condition_for(DATASET_ID_LLM_BANK) == DatasetMetrics(
        benchmark=Benchmark.LLM,
        required=Metric.INSTRUCTION_FOLLOWING,
        local=frozenset({Metric.TTFT}),
    )


async def _fetch_llm(
    writer: MagicMock,
    values: list[dict[str, Any]],
    *,
    persona_conditions: dict[str, Condition] | None = None,
) -> tuple[RunStatus, int]:
    list_json = _list_json(
        {"run_id": "R1", "create_time": _iso(timedelta(hours=1)), "persona_id": "P1"}
    )
    async with _fake_client(list_json, _run_json(values, metric_id="IID")) as client:
        return await fetch_v2v._fetch_one_provider(
            client,
            writer,
            spec=LLM_SPEC,
            agent_id="a1",
            metric_ids={Metric.INSTRUCTION_FOLLOWING: "IID"},
            test_set_id="TSD",
            persona_conditions=persona_conditions,
            period_seconds=10_800,
            stale_grace_seconds=5_400,
        )


@pytest.mark.asyncio
async def test_text_agent_uses_instruction_as_the_clean_bank_anchor() -> None:
    writer = _stub_writer()
    writer.conversation_ttft = AsyncMock(return_value={"s1": 0.4126})
    values = [
        {"simulation_output_id": "s1", "value": "YES"},
        {"simulation_output_id": "s2", "value": "NO"},
    ]

    with capture_logs() as logs:
        status, ingested = await _fetch_llm(writer, values)

    assert (status, ingested) == (RunStatus.SUCCEEDED, 1)
    assert writer.start_run.await_args.kwargs["dataset_id"] == DATASET_ID_LLM_BANK
    writer.conversation_ttft.assert_awaited_once_with(["s1", "s2"])
    rows = writer.record_results.await_args.args[0]
    assert {(r.metric_type, r.audio_filename, r.metric_value, r.metric_units) for r in rows} == {
        (Metric.INSTRUCTION_FOLLOWING, "R1/s1", 100.0, "percent"),
        (Metric.INSTRUCTION_FOLLOWING, "R1/s2", 0.0, "percent"),
        (Metric.TTFT, "R1/s1", 0.413, "seconds"),
    }
    assert all(r.benchmark is Benchmark.LLM for r in rows)
    gaps = [log for log in logs if log["event"] == "local_metric_coverage_gap"]
    assert [(log["metric"], log["conversations"]) for log in gaps] == [("TTFT", 1)]


def test_local_metrics_are_ingestable_only_with_a_source() -> None:
    condition = DatasetMetrics(
        required=Metric.INSTRUCTION_FOLLOWING, local=frozenset({Metric.TTFT, Metric.V2V})
    )
    assert fetch_v2v._ingestable(condition, {Metric.INSTRUCTION_FOLLOWING: "IID"}) == {
        Metric.INSTRUCTION_FOLLOWING,
        Metric.TTFT,
    }


@pytest.mark.asyncio
async def test_ingest_run_defaults_to_writing_local_metrics() -> None:
    writer = _stub_writer()
    writer.conversation_ttft = AsyncMock(return_value={"s1": 0.25})
    values = [{"simulation_output_id": "s1", "value": "YES"}]
    async with _fake_client({}, _run_json(values, metric_id="IID")) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=LLM_SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids={Metric.INSTRUCTION_FOLLOWING: "IID"},
            condition=condition_for(DATASET_ID_LLM_BANK),
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    assert {r.metric_type for r in rows} == {Metric.INSTRUCTION_FOLLOWING, Metric.TTFT}


@pytest.mark.asyncio
async def test_ttft_backfills_onto_an_ingested_run_once_turns_exist() -> None:
    writer = _stub_writer()
    writer.coval_metric_ingested = AsyncMock(
        side_effect=lambda **kw: kw["metric_type"] is Metric.INSTRUCTION_FOLLOWING
    )
    values = [{"simulation_output_id": "s1", "value": "YES"}]

    status, ingested = await _fetch_llm(writer, values)
    assert (status, ingested) == (RunStatus.SUCCEEDED, 0)
    writer.start_run.assert_not_awaited()

    writer.conversation_ttft = AsyncMock(return_value={"s1": 0.2})
    status, ingested = await _fetch_llm(writer, values)
    assert (status, ingested) == (RunStatus.SUCCEEDED, 1)
    rows = writer.record_results.await_args.args[0]
    assert [(r.metric_type, r.metric_value) for r in rows] == [(Metric.TTFT, 0.2)]


@pytest.mark.asyncio
async def test_non_clean_text_personas_are_not_ingested() -> None:
    writer = _stub_writer()
    values = [{"simulation_output_id": "s1", "value": "YES"}]

    status, ingested = await _fetch_llm(writer, values, persona_conditions={"P1": Condition.NOISY})

    assert (status, ingested) == (RunStatus.FAILED, 0)
    writer.coval_metric_ingested.assert_not_awaited()
    writer.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_and_write_llm_names_the_missing_suite_setting() -> None:
    with pytest.raises(RuntimeError, match="coval_s2s_bank_persona_id"):
        await fetch_v2v.fetch_and_write_v2v(
            Settings(coval_s2s_bank_test_set_id="TSB", coval_s2s_bank_instruction_metric_id="IID"),
            benchmark=Benchmark.LLM,
            llm_agent_ids={"phonely": "a1"},
        )


@pytest.mark.asyncio
async def test_fetch_and_write_v2v_still_requires_the_latency_metric_for_legacy_agents() -> None:
    settings = Settings(coval_s2s_openai_agent_id="a1", coval_s2s_dental_test_set_id="TSD")

    with pytest.raises(RuntimeError, match="coval_s2s_latency_metric_id is not set"):
        await fetch_v2v.fetch_and_write_v2v(settings)


@pytest.mark.asyncio
async def test_fetch_one_provider_rejects_a_dataset_from_another_benchmark() -> None:
    writer = _stub_writer()
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})

    with capture_logs() as logs:
        async with _fake_client(list_json, {}) as client:
            status, ingested = await fetch_v2v._fetch_one_provider(
                client,
                writer,
                spec=replace(LLM_SPEC, family=FAMILY_MULTITURN),
                agent_id="a1",
                metric_ids={Metric.INSTRUCTION_FOLLOWING: "IID"},
                test_set_id="TS1",
                period_seconds=10_800,
                stale_grace_seconds=5_400,
            )

    assert (status, ingested) == (RunStatus.FAILED, 0)
    assert any(
        log["event"] == "provider_fetch_failed" and "belongs to S2S, not LLM" in log["error"]
        for log in logs
    )
    writer.coval_metric_ingested.assert_not_awaited()


@pytest.mark.asyncio
async def test_samples_publish_without_the_shared_test_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dental-only deployment still publishes: the gate reads the run's own set.

    ``coval_s2s_test_set_id`` is deliberately unset here. Gating publication on it
    would fill ``sampled_runs`` and then silently never ship them, leaving the
    public sample card frozen while ingestion looked healthy.
    """
    monkeypatch.delenv("COVAL_S2S_GEMINI_AGENT_ID", raising=False)
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_dental_test_set_id="TSD",
        s2s_samples_bucket="bucket",
    )
    assert settings.coval_s2s_test_set_id is None

    writer = _stub_writer()
    list_json = _list_json({"run_id": "R1", "create_time": _iso(timedelta(hours=1))})
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    client = _fake_client(list_json, _run_json(values))

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    publish = AsyncMock(return_value=1)
    monkeypatch.setattr(fetch_v2v, "_client", lambda _s: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)
    monkeypatch.setattr(fetch_v2v, "publish_tick_sample", publish)

    await fetch_v2v.fetch_and_write_v2v(settings)

    publish.assert_awaited_once()
    assert publish.await_args is not None
    assert publish.await_args.kwargs["test_set_id"] == "TSD"
    assert publish.await_args.kwargs["dataset_id"] == DATASET_ID_DENTAL


@pytest.mark.asyncio
async def test_each_dataset_publishes_its_own_sample_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dental and the bank set never share a tick: each partition gets its own
    publish with only its own runs and only its own expected models."""
    monkeypatch.delenv("COVAL_S2S_GEMINI_AGENT_ID", raising=False)
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_dental_test_set_id="TSD",
        coval_s2s_scenarios={
            "bank": ScenarioCovalIds(
                test_set_id="TSB", agents={"gpt-realtime": "b1"}, personas={"clean": "PID"}
            )
        },
        coval_s2s_bank_test_set_id="TSB",
        coval_s2s_bank_instruction_metric_id="BIM",
        s2s_samples_bucket="bucket",
    )

    writer = _stub_writer()
    list_json = _list_json(
        {"run_id": "R1", "create_time": _iso(timedelta(hours=1)), "persona_id": "PID"}
    )
    run_json = {
        "run": {
            "error_status": "SUCCESS",
            "results": {
                "metrics": {
                    "MID": {"values": [{"simulation_output_id": "s1", "value": 0.5}]},
                    "BIM": {"values": [{"simulation_output_id": "s1", "value": 0.75}]},
                }
            },
        }
    }
    client = _fake_client(list_json, run_json)

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    publish = AsyncMock(return_value=1)
    monkeypatch.setattr(fetch_v2v, "_client", lambda _s: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)
    monkeypatch.setattr(fetch_v2v, "publish_tick_sample", publish)

    await fetch_v2v.fetch_and_write_v2v(settings)

    calls = {c.kwargs["dataset_id"]: c.kwargs for c in publish.await_args_list}
    assert set(calls) == {DATASET_ID_DENTAL, DATASET_ID_BANK}
    dental, bank = calls[DATASET_ID_DENTAL], calls[DATASET_ID_BANK]
    assert dental["test_set_id"] == "TSD"
    assert {r.key for r in dental["runs"]} == {("openai", "gpt-realtime")}
    assert dental["expected_models"] == {("openai", "gpt-realtime")}
    assert bank["test_set_id"] == "TSB"
    assert all(r.dataset_id == DATASET_ID_BANK for r in bank["runs"])
    assert bank["expected_models"] == {("openai", "gpt-realtime")}
    # The bank run's judge fraction landed as a percentage under the instruction metric.
    rows = [r for call in writer.record_results.await_args_list for r in call.args[0]]
    assert [r.metric_value for r in rows if r.metric_type == "InstructionFollowing"] == [75.0]


@pytest.mark.asyncio
async def test_fetch_and_write_v2v_noop_skips_matview_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COVAL_S2S_GEMINI_AGENT_ID", raising=False)
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_dental_test_set_id="TSD",
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

    assert statuses == {"s2s-dental:openai:gpt-realtime": RunStatus.SUCCEEDED}
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
        agent_id="g1",
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


def test_import_identity_hashes_coval_test_set_provenance() -> None:
    identity = fetch_v2v._import_identity(
        spec=SPEC,
        coval_run=CovalRun(run_id="external", create_time=None),
        dataset_id="s2s-multiturn-v1",
        dataset_sha256="8ef475f7-5b57-46db-939e-45b10486cba8",
        workspace_id=None,
    )

    assert (
        identity.dataset_sha256
        == hashlib.sha256(b"8ef475f7-5b57-46db-939e-45b10486cba8").hexdigest()
    )


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
    with pytest.raises(fetch_v2v.UnmappedPersona, match="no condition mapped"):
        fetch_v2v._dataset_identity("TS1", "PUNKNOWN", persona_conditions=personas)


@pytest.mark.asyncio
async def test_unmapped_persona_skips_only_its_run() -> None:
    """One stray persona must not fail the provider or hide the mapped runs."""
    writer = _stub_writer()
    list_json = _list_json(
        {"run_id": "R1", "create_time": _iso(timedelta(hours=1)), "persona_id": "PC"},
        {"run_id": "R2", "create_time": _iso(timedelta(hours=2)), "persona_id": "PUNKNOWN"},
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]
    unmapped: dict[str, int] = {}
    with capture_logs() as logs:
        async with _fake_client(list_json, _run_json(values)) as client:
            status, ingested = await fetch_v2v._fetch_one_provider(
                client,
                writer,
                spec=SPEC,
                agent_id="a1",
                metric_ids=LATENCY_IDS,
                test_set_id="TS1",
                persona_conditions={"PC": Condition.CLEAN},
                period_seconds=10_800,
                stale_grace_seconds=5_400,
                unmapped_personas=unmapped,
            )
    assert (status, ingested) == (RunStatus.SUCCEEDED, 1)
    assert unmapped == {"PUNKNOWN": 1}
    assert [
        (log["coval_run_id"], log["persona_id"])
        for log in logs
        if log["event"] == "run_persona_unmapped"
    ] == [("R2", "PUNKNOWN")]
    assert not any(log["event"] == "provider_fetch_failed" for log in logs)


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
            period_seconds=10_800,
            stale_grace_seconds=5_400,
            sampled_runs=sampled,
        )
    assert [r.coval_run_id for r in sampled] == ["RCLEAN"]


@pytest.mark.asyncio
async def test_fetch_and_write_rejects_noisy_persona_without_a_test_set() -> None:
    # Without a test set the persona split can never fire, so fail loudly.
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_dental_test_set_id="TSD",
        coval_s2s_noisy_persona_id="PN",
    )
    with pytest.raises(RuntimeError, match="requires coval_s2s_test_set_id"):
        await fetch_v2v.fetch_and_write_v2v(settings)


@pytest.mark.asyncio
async def test_fetch_and_write_rejects_a_blank_happypath_test_set() -> None:
    # Blank would skip its agents with only a warning, indistinguishable from unset.
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_happypath_test_set_id="   ",
    )
    with pytest.raises(RuntimeError, match="coval_s2s_happypath_test_set_id must not be blank"):
        await fetch_v2v.fetch_and_write_v2v(settings)


@pytest.mark.asyncio
async def test_fetch_and_write_rejects_a_blank_dental_test_set() -> None:
    # gray/red read this one, so blank would strand both with only a warning.
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_dental_test_set_id="   ",
    )
    with pytest.raises(RuntimeError, match="coval_s2s_dental_test_set_id must not be blank"):
        await fetch_v2v.fetch_and_write_v2v(settings)


@pytest.mark.asyncio
async def test_fetch_and_write_requires_a_dental_test_set_for_a_dental_agent() -> None:
    # Unset skipped the agent with only a warning, which is how gray and red
    # went five days without ingesting a row.
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_gray_agent_id="a2",
    )
    with pytest.raises(RuntimeError, match="coval_s2s_dental_test_set_id is required"):
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
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_test_set_id="TS1",
        coval_s2s_dental_test_set_id="TSD",
        coval_s2s_instruction_metric_id="IID",
        coval_s2s_interruption_metric_id="RID",
    )
    with pytest.raises(RuntimeError, match="no _VALUE_MAPPERS entry"):
        await fetch_v2v.fetch_and_write_v2v(settings)


def test_instruction_value_scales_a_judge_fraction_to_percent() -> None:
    assert fetch_v2v._instruction_value(0.75) == (75.0, ResultStatus.SUCCESS)
    assert fetch_v2v._instruction_value(1) == (100.0, ResultStatus.SUCCESS)
    assert fetch_v2v._instruction_value(0.0) == (0.0, ResultStatus.SUCCESS)
    assert fetch_v2v._instruction_value(0.3333333333333333) == (
        33.33333333333333,
        ResultStatus.SUCCESS,
    )
    assert fetch_v2v._instruction_value("YES") == (100.0, ResultStatus.SUCCESS)
    assert fetch_v2v._instruction_value("UNKNOWN") is None
    with pytest.raises(fetch_v2v.InvalidInstructionVerdict):
        fetch_v2v._instruction_value(1.5)
    with pytest.raises(fetch_v2v.InvalidInstructionVerdict):
        fetch_v2v._instruction_value(True)


def test_latency_metric_is_required_only_for_a_family_that_fetches_it() -> None:
    assert fetch_v2v._fetches_v2v(FAMILY_BANK)
    assert fetch_v2v._fetches_v2v(FAMILY_DENTAL)
    assert not fetch_v2v._fetches_v2v(FAMILY_LLM_BANK)


def test_interruption_value_maps() -> None:
    assert fetch_v2v._interruption_value(1.1666666666666632) == (
        1.1666666666666632,
        ResultStatus.SUCCESS,
    )
    assert fetch_v2v._interruption_value(0.0041) == (0.0041, ResultStatus.SUCCESS)
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=ALL_IDS,
            condition=condition_for(DATASET_ID_MULTITURN),
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


def test_call_length_value_maps() -> None:
    assert fetch_v2v._call_length_value(73.4) == (73.4, ResultStatus.SUCCESS)
    assert fetch_v2v._call_length_value(0) == (0.0, ResultStatus.SUCCESS)
    assert fetch_v2v._call_length_value(None) == (None, ResultStatus.FAILED)
    assert fetch_v2v._call_length_value("73.4") == (None, ResultStatus.FAILED)
    assert fetch_v2v._call_length_value(True) == (None, ResultStatus.FAILED)


@pytest.mark.asyncio
async def test_ingest_run_writes_call_length_rows() -> None:
    writer = _stub_writer()
    instruction = [{"simulation_output_id": f"s{i}", "value": "YES"} for i in range(2)]
    call_length = [
        {"simulation_output_id": "s0", "value": 48.9},
        {"simulation_output_id": "s1", "value": None},
    ]
    run_json = {
        "run": {
            "error_status": "SUCCESS",
            "results": {
                "metrics": {"IID": {"values": instruction}, "CID": {"values": call_length}}
            },
        }
    }
    async with _fake_client({}, run_json) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids={Metric.INSTRUCTION_FOLLOWING: "IID", Metric.CALL_LENGTH: "CID"},
            condition=condition_for(DATASET_ID_BANK_LOW),
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    lengths = [r for r in rows if r.metric_type == Metric.CALL_LENGTH]
    assert [(r.audio_filename, r.metric_value, r.status) for r in lengths] == [
        ("R1/s0", 48.9, ResultStatus.SUCCESS),
        ("R1/s1", None, ResultStatus.FAILED),
    ]
    assert all(r.metric_units == "seconds" for r in lengths)


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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED  # latency intact
    rows = writer.record_results.await_args.args[0]
    # s1 is unscored and s2 unmeasured, so only s0 is covered by both.
    instruction_rows = [r for r in rows if r.metric_type == Metric.INSTRUCTION_FOLLOWING]
    assert len(instruction_rows) == 1
    assert instruction_rows[0].audio_filename.endswith("s0")


@pytest.mark.asyncio
async def test_ingest_run_extra_ids_are_trimmed_not_dropped() -> None:
    # A call the anchor never measured (agent silent, so no turn gap) still gets a
    # judge verdict. That one conversation is trimmed; the rest are kept.
    writer = _stub_writer()
    latency = [
        {"simulation_output_id": "s0", "value": 0.5},
        {"simulation_output_id": "s1", "value": 0.5},
    ]
    instruction = [
        {"simulation_output_id": "s0", "value": "YES"},
        {"simulation_output_id": "s1", "value": "NO"},
        {"simulation_output_id": "s2", "value": "YES"},  # unmeasured by the anchor
    ]
    async with _fake_client({}, _multi_metric_run(latency, instruction)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    instruction_rows = [r for r in rows if r.metric_type == Metric.INSTRUCTION_FOLLOWING]
    assert len(instruction_rows) == 2
    assert not any(r.audio_filename.endswith("s2") for r in instruction_rows)


@pytest.mark.asyncio
async def test_ingest_run_drops_id_less_values_before_trimming() -> None:
    # Two values with no simulation_output_id are not the same conversation, so
    # neither may be matched against the other.
    writer = _stub_writer()
    latency: list[dict[str, Any]] = [
        {"simulation_output_id": "s0", "value": 0.5},
        {"value": 0.5},
    ]
    instruction: list[dict[str, Any]] = [
        {"simulation_output_id": "s0", "value": "YES"},
        {"value": "NO"},
    ]
    async with _fake_client({}, _multi_metric_run(latency, instruction)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    instruction_rows = [r for r in rows if r.metric_type == Metric.INSTRUCTION_FOLLOWING]
    assert len(instruction_rows) == 1
    assert instruction_rows[0].audio_filename.endswith("s0")


@pytest.mark.asyncio
async def test_ingest_run_duplicate_ids_still_drop_the_metric() -> None:
    writer = _stub_writer()
    latency = [
        {"simulation_output_id": "s0", "value": 0.5},
        {"simulation_output_id": "s1", "value": 0.5},
    ]
    instruction = [
        {"simulation_output_id": "s0", "value": "YES"},
        {"simulation_output_id": "s0", "value": "NO"},
        {"simulation_output_id": "s1", "value": "YES"},
    ]
    async with _fake_client({}, _multi_metric_run(latency, instruction)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED
    rows = writer.record_results.await_args.args[0]
    assert all(r.metric_type == Metric.V2V for r in rows)


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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            period_seconds=10_800,
        )
    assert status is RunStatus.SUCCEEDED  # latency kept
    rows = writer.record_results.await_args.args[0]
    assert all(r.metric_type == Metric.V2V for r in rows)  # instruction discarded


@pytest.mark.asyncio
async def test_fetch_and_write_requires_id_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    # instruction id set but test-set id missing -> startup failure (must be paired).
    settings = Settings(
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            pending=frozenset({Metric.INSTRUCTION_FOLLOWING}),
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            pending=frozenset({Metric.INSTRUCTION_FOLLOWING}),
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN_NOISY),
            dataset_id=DATASET_ID_MULTITURN_NOISY,
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN),
            dataset_id=DATASET_ID_MULTITURN,
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN_NOISY),
            dataset_id=DATASET_ID_MULTITURN_NOISY,
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN_NOISY),
            dataset_id=DATASET_ID_MULTITURN_NOISY,
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
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
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=LATENCY_IDS,
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


def test_log_run_unmapped_persona_emits_its_event_with_the_fix() -> None:
    with capture_logs() as logs:
        log_run_unmapped_persona({"PB": 4, "PA": 1})
    assert [entry["event"] for entry in logs] == ["RUN_UNMAPPED_PERSONA"]
    assert logs[0]["error"] == (
        "runs skipped for unmapped personas: PA (1), PB (4); "
        "map or skip them in coval_s2s_condition_personas"
    )


def test_log_run_failed_emits_run_failed_event() -> None:
    # Unchanged contract shared with the STT/TTS orchestrator's failure metric.
    with capture_logs() as logs:
        log_run_failed("s2s fetch failed for all providers")
    assert [entry["event"] for entry in logs] == ["RUN_FAILED"]


def _run_fetch_cli(
    monkeypatch: pytest.MonkeyPatch,
    statuses: dict[str, RunStatus],
) -> tuple[int, list[str]]:
    async def fake_fetch(_settings: Settings, **kwargs: object) -> dict[str, RunStatus]:
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


@pytest.mark.asyncio
async def test_ingest_run_dual_writes_one_observation_per_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _stub_writer()
    dual_write = AsyncMock()
    monkeypatch.setattr("coval_bench.runner.normalized.dual_write", dual_write)
    latency = [
        {"simulation_output_id": "s1", "value": 0.5},
        {"simulation_output_id": "s2", "value": 0.6},
    ]
    instruction = [
        {"simulation_output_id": "s1", "value": "YES"},
        {"simulation_output_id": "s2", "value": "NO"},
    ]
    fixture = {
        "run": {
            "results": {"metrics": {"MID": {"values": latency}, "IID": {"values": instruction}}}
        }
    }

    async with _fake_client({}, fixture) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=IDS,
            condition=condition_for(DATASET_ID_MULTITURN),
            dataset_id=DATASET_ID_MULTITURN,
            dataset_sha256="test-set:persona",
            period_seconds=10_800,
            normalized_dual_write_enabled=True,
        )

    assert status is RunStatus.SUCCEEDED
    assert dual_write.await_count == 2
    calls = {call.kwargs["sample_id"]: call.kwargs for call in dual_write.await_args_list}
    assert set(calls) == {"R1/s1", "R1/s2"}
    assert {row.metric_type for row in calls["R1/s1"]["results"]} == {
        Metric.V2V,
        Metric.INSTRUCTION_FOLLOWING,
    }
    assert calls["R1/s1"]["dataset_sha256"] == hashlib.sha256(b"test-set:persona").hexdigest()
    assert calls["R1/s1"]["executor"] is MetricExecutor.COVAL_API
    assert calls["R1/s1"]["captured_at"] == writer.record_results.await_args.kwargs["created_at"]
    for kwargs in calls.values():
        assert kwargs["db_retry_attempts"] == 3
        assert not any("semaphore" in key for key in kwargs)


@pytest.mark.asyncio
async def test_ingest_run_dual_writes_llm_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _stub_writer()
    dual_write = AsyncMock()
    monkeypatch.setattr("coval_bench.runner.normalized.dual_write", dual_write)
    condition = DatasetMetrics(
        benchmark=Benchmark.LLM,
        required=Metric.INSTRUCTION_FOLLOWING,
    )

    async with _fake_client(
        {}, _run_json([{"simulation_output_id": "s1", "value": "YES"}], metric_id="IID")
    ) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=LLM_SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids={Metric.INSTRUCTION_FOLLOWING: "IID"},
            condition=condition,
            dataset_id="llm-bank-v1",
            period_seconds=10_800,
            normalized_dual_write_enabled=True,
        )

    assert status is RunStatus.SUCCEEDED
    writer.record_results.assert_awaited_once()
    rows = writer.record_results.await_args.args[0]
    assert len(rows) == 1
    assert rows[0].benchmark is Benchmark.LLM
    dual_write.assert_awaited_once()
    kwargs = dual_write.await_args_list[0].kwargs
    assert (kwargs["benchmark"], kwargs["sample_id"]) == (Benchmark.LLM, "R1/s1")


@pytest.mark.asyncio
async def test_ingest_run_normalized_failure_does_not_lose_legacy_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _stub_writer()
    dual_write = AsyncMock(side_effect=RuntimeError("normalized unavailable"))
    monkeypatch.setattr("coval_bench.runner.normalized.dual_write", dual_write)
    values = [{"simulation_output_id": "s1", "value": 0.5}]

    async with _fake_client({}, _run_json(values)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=CovalRun(run_id="R1", create_time=None),
            metric_ids=LATENCY_IDS,
            dataset_sha256="f" * 64,
            period_seconds=10_800,
            normalized_dual_write_enabled=True,
        )

    assert status is RunStatus.SUCCEEDED
    writer.record_results.assert_awaited_once()
    writer.finish_run.assert_awaited_once()
    dual_write.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spec", "condition", "metric_ids"),
    [
        (SPEC, DEFAULT_CONDITION, LATENCY_IDS),
        (
            LLM_SPEC,
            DatasetMetrics(
                benchmark=Benchmark.LLM,
                required=Metric.INSTRUCTION_FOLLOWING,
            ),
            {Metric.INSTRUCTION_FOLLOWING: "IID"},
        ),
    ],
)
async def test_required_s2s_and_llm_capture_precedes_legacy_and_marks_backlog_partial(
    monkeypatch: pytest.MonkeyPatch,
    spec: AgentSpec,
    condition: DatasetMetrics,
    metric_ids: dict[Metric, str],
) -> None:
    writer = _stub_writer()
    writer.finish_run_exact = AsyncMock()
    persist = AsyncMock(return_value="pending")
    upload_state = MagicMock(return_value=("gs://private/state", "a" * 64))
    monkeypatch.setattr("coval_bench.runner.normalized.persist_capture", persist)
    monkeypatch.setattr("coval_bench.runner.capture.upload_run_state", upload_state)
    monkeypatch.setattr(fetch_v2v, "upload_import_run_claim", lambda _c, _b, claim: claim)
    metric_id = next(iter(metric_ids.values()))
    metric_value: object = "YES" if spec.benchmark is Benchmark.LLM else 0.5
    coval_run = CovalRun(run_id="R1", create_time=None)
    import_identity = fetch_v2v._import_identity(
        spec=spec,
        coval_run=coval_run,
        dataset_id="required-v1",
        dataset_sha256="f" * 64,
        workspace_id=None,
    )

    async with _fake_client(
        {}, _run_json([{"simulation_output_id": "s1", "value": metric_value}], metric_id)
    ) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=spec,
            coval_run=coval_run,
            metric_ids=metric_ids,
            condition=condition,
            dataset_id="required-v1",
            dataset_sha256="f" * 64,
            period_seconds=10_800,
            normalized_dual_write_enabled=True,
            normalized_capture_required=True,
            artifact_client=object(),
            artifact_bucket="private",
            import_identity=import_identity,
            import_generation=0,
        )

    assert status is RunStatus.PARTIAL
    writer.record_results.assert_not_awaited()
    writer.start_run.assert_not_awaited()
    writer.reserve_run_id.assert_awaited_once()
    writer.ensure_capture_run.assert_awaited_once()
    persist.assert_awaited_once()
    writer.finish_run_exact.assert_awaited_once()
    assert writer.finish_run_exact.await_args.kwargs["status"] is RunStatus.PARTIAL
    assert writer.finish_run_exact.await_args.kwargs["error"] == "normalized capture pending"
    assert upload_state.call_args_list[0].args[3] == "manifest"
    assert upload_state.call_args_list[1].args[3] == "seal"


@pytest.mark.asyncio
async def test_required_import_restart_reuses_durable_claim_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _stub_writer()
    writer.finish_run_exact = AsyncMock()
    persist = AsyncMock(return_value="completed")
    upload_state = MagicMock(return_value=("gs://private/state", "a" * 64))
    monkeypatch.setattr("coval_bench.runner.normalized.persist_capture", persist)
    monkeypatch.setattr("coval_bench.runner.capture.upload_run_state", upload_state)
    coval_run = CovalRun(run_id="external-run", create_time=datetime(2026, 9, 17, tzinfo=UTC))
    identity = fetch_v2v._import_identity(
        spec=SPEC,
        coval_run=coval_run,
        dataset_id="required-v1",
        dataset_sha256="f" * 64,
        workspace_id="workspace",
    )
    claim = ImportRunClaim(
        identity=identity,
        generation=0,
        run_id=41,
        started_at=datetime(2026, 9, 17, tzinfo=UTC),
        scheduled_at=datetime(2026, 9, 17, tzinfo=UTC),
        captured_at=datetime(2026, 9, 17, tzinfo=UTC),
        metric_types=[str(Metric.V2V)],
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]

    async with _fake_client({}, _run_json(values)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=coval_run,
            metric_ids=LATENCY_IDS,
            dataset_id="required-v1",
            dataset_sha256="f" * 64,
            period_seconds=10_800,
            normalized_capture_required=True,
            artifact_client=object(),
            artifact_bucket="private",
            workspace_id="workspace",
            import_identity=identity,
            import_generation=0,
            import_claim=claim,
        )

    assert status is RunStatus.SUCCEEDED
    writer.start_run.assert_not_awaited()
    writer.reserve_run_id.assert_not_awaited()
    writer.ensure_capture_run.assert_awaited_once()
    assert writer.ensure_capture_run.await_args.args[0] == 41
    persist.assert_awaited_once()


@pytest.mark.asyncio
async def test_required_import_adopts_concurrent_seal_without_failing_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _stub_writer()
    writer.finish_run_exact = AsyncMock()
    persist = AsyncMock(return_value="completed")
    monkeypatch.setattr("coval_bench.runner.normalized.persist_capture", persist)
    coval_run = CovalRun(run_id="external-run", create_time=datetime(2026, 9, 17, tzinfo=UTC))
    identity = fetch_v2v._import_identity(
        spec=SPEC,
        coval_run=coval_run,
        dataset_id="required-v1",
        dataset_sha256="f" * 64,
        workspace_id="workspace",
    )
    started_at = datetime(2026, 9, 17, tzinfo=UTC)
    claim = ImportRunClaim(
        identity=identity,
        generation=0,
        run_id=41,
        started_at=started_at,
        scheduled_at=started_at,
        captured_at=started_at,
        metric_types=[str(Metric.V2V)],
    )
    capture_id = identity_digest(
        build_capture_identity(
            run_id=41,
            benchmark="S2S",
            dataset_id="required-v1",
            sample_id="external-run/s1",
            provider=SPEC.provider,
            model=SPEC.model,
            capture_id="external-run:external-run/s1",
        )
    )
    winner_finished_at = started_at + timedelta(minutes=1)
    winner_seal = RunSeal(
        run_id=41,
        intended_status=str(RunStatus.SUCCEEDED),
        stored_status=str(RunStatus.SUCCEEDED),
        finished_at=winner_finished_at,
        expected_capture_ids=[capture_id],
    )
    seal_calls = 0

    def upload_state(
        _client: object, _bucket: str, _run_id: int, kind: str, value: object
    ) -> tuple[str, str]:
        nonlocal seal_calls
        if kind == "seal":
            seal_calls += 1
            if seal_calls == 1:
                raise ValueError("concurrent seal")
            assert value == winner_seal
        return f"gs://private/{kind}", "a" * 64

    monkeypatch.setattr("coval_bench.runner.capture.upload_run_state", upload_state)
    monkeypatch.setattr(
        fetch_v2v,
        "read_run_state",
        lambda _client, _bucket, _run_id, kind: (
            winner_seal.model_dump(mode="json") if kind == "seal" else None
        ),
    )
    values = [{"simulation_output_id": "s1", "value": 0.5}]

    async with _fake_client({}, _run_json(values)) as client:
        status = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=coval_run,
            metric_ids=LATENCY_IDS,
            dataset_id="required-v1",
            dataset_sha256="f" * 64,
            period_seconds=10_800,
            normalized_capture_required=True,
            artifact_client=object(),
            artifact_bucket="private",
            workspace_id="workspace",
            import_identity=identity,
            import_generation=0,
            import_claim=claim,
        )

    assert status is RunStatus.SUCCEEDED
    writer.finish_run.assert_not_awaited()
    writer.finish_run_exact.assert_awaited_once()
    assert writer.finish_run_exact.await_args.kwargs["finished_at"] == winner_finished_at
    assert writer.finish_run_exact.await_args.kwargs["status"] is RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_required_import_manifest_failure_remains_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _stub_writer()
    writer.finish_run_exact = AsyncMock()
    persist = AsyncMock(return_value="completed")
    monkeypatch.setattr("coval_bench.runner.normalized.persist_capture", persist)
    coval_run = CovalRun(run_id="external-run", create_time=datetime(2026, 9, 17, tzinfo=UTC))
    identity = fetch_v2v._import_identity(
        spec=SPEC,
        coval_run=coval_run,
        dataset_id="required-v1",
        dataset_sha256="f" * 64,
        workspace_id="workspace",
    )
    started_at = datetime(2026, 9, 17, tzinfo=UTC)
    claim = ImportRunClaim(
        identity=identity,
        generation=0,
        run_id=41,
        started_at=started_at,
        scheduled_at=started_at,
        captured_at=started_at,
        metric_types=[str(Metric.V2V)],
    )
    upload_state = MagicMock(side_effect=OSError("manifest unavailable"))
    monkeypatch.setattr("coval_bench.runner.capture.upload_run_state", upload_state)
    values = [{"simulation_output_id": "s1", "value": 0.5}]

    async with _fake_client({}, _run_json(values)) as client:
        first = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=coval_run,
            metric_ids=LATENCY_IDS,
            dataset_id="required-v1",
            dataset_sha256="f" * 64,
            period_seconds=10_800,
            normalized_capture_required=True,
            artifact_client=object(),
            artifact_bucket="private",
            workspace_id="workspace",
            import_identity=identity,
            import_generation=0,
            import_claim=claim,
        )

    assert first is RunStatus.PARTIAL
    writer.finish_run.assert_not_awaited()
    upload_state.side_effect = None
    upload_state.return_value = ("gs://private/state", "a" * 64)

    async with _fake_client({}, _run_json(values)) as client:
        second = await fetch_v2v._ingest_run(
            client,
            writer,
            spec=SPEC,
            coval_run=coval_run,
            metric_ids=LATENCY_IDS,
            dataset_id="required-v1",
            dataset_sha256="f" * 64,
            period_seconds=10_800,
            normalized_capture_required=True,
            artifact_client=object(),
            artifact_bucket="private",
            workspace_id="workspace",
            import_identity=identity,
            import_generation=0,
            import_claim=claim,
        )

    assert second is RunStatus.SUCCEEDED
    writer.finish_run.assert_not_awaited()
    writer.ensure_capture_run.assert_awaited()
    assert {call.args[0] for call in writer.ensure_capture_run.await_args_list} == {41}


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_failure", [False, True])
async def test_fetch_and_write_v2v_propagates_normalized_gate(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_failure: bool,
) -> None:
    monkeypatch.delenv("COVAL_S2S_GEMINI_AGENT_ID", raising=False)
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_openai_agent_id="a1",
        coval_s2s_dental_test_set_id="TSD",
        normalized_dual_write_enabled=True,
        benchmark_artifact_bucket="private-artifacts",
    )
    client = _fake_client({}, {})
    writer = _stub_writer()
    if snapshot_failure:
        writer.refresh_dashboard_summaries.side_effect = RuntimeError("snapshot unavailable")

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    fetch_one = AsyncMock(return_value=(RunStatus.SUCCEEDED, 0))
    monkeypatch.setattr(fetch_v2v, "_client", lambda _settings: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)
    monkeypatch.setattr(fetch_v2v, "_fetch_one_provider", fetch_one)

    statuses = await fetch_v2v.fetch_and_write_v2v(settings)

    writer.refresh_dashboard_summaries.assert_awaited_once_with()
    assert statuses == {"s2s-dental:openai:gpt-realtime": RunStatus.SUCCEEDED}
    assert fetch_one.await_args is not None
    assert fetch_one.await_args.kwargs["normalized_dual_write_enabled"] is True


def test_scenario_personas_join_the_condition_map() -> None:
    settings = Settings(
        coval_s2s_condition_personas={"legacy": "noisy"},
        coval_s2s_scenarios={
            "happy-smile": ScenarioCovalIds(
                test_set_id="TSS", personas={"clean": "p-clean", "hard": "p-hard"}
            )
        },
    )
    assert fetch_v2v._all_persona_conditions(settings) == {
        "legacy": "noisy",
        "p-clean": "clean",
        "p-hard": "hard",
    }
    assert fetch_v2v._all_persona_labels(settings) == {"p-clean": "Clean", "p-hard": "Poor cell"}


def test_the_same_persona_may_serve_several_scenarios() -> None:
    settings = Settings(
        coval_s2s_condition_personas={"p1": "clean"},
        coval_s2s_scenarios={
            "bank": ScenarioCovalIds(test_set_id="TSB", personas={"clean": "p1"}),
            "happy-smile": ScenarioCovalIds(test_set_id="TSS", personas={"clean": "p1"}),
        },
    )
    assert fetch_v2v._all_persona_conditions(settings) == {"p1": "clean"}


def test_a_persona_with_two_conditions_is_a_configuration_error() -> None:
    settings = Settings(
        coval_s2s_condition_personas={"p1": "clean"},
        coval_s2s_scenarios={"bank": ScenarioCovalIds(test_set_id="TSB", personas={"low": "p1"})},
    )
    with pytest.raises(RuntimeError, match="'clean' in coval_s2s_condition_personas but 'low'"):
        fetch_v2v._all_persona_conditions(settings)


def test_a_scenario_persona_with_an_unknown_condition_is_rejected_at_load() -> None:
    with pytest.raises(ValueError, match="persona condition"):
        ScenarioCovalIds(test_set_id="TSB", personas={"noisy": "p1"})


def test_an_unknown_scenario_slug_is_rejected_at_load() -> None:
    with pytest.raises(ValueError, match="happy_smile"):
        Settings(coval_s2s_scenarios={"happy_smile": ScenarioCovalIds(test_set_id="TSS")})


def test_an_unknown_agent_model_key_faults() -> None:
    settings = Settings(
        coval_s2s_scenarios={
            "bank": ScenarioCovalIds(
                test_set_id="TSB", agents={"gpt-realtime-1": "b1"}, personas={"clean": "p"}
            )
        }
    )
    with pytest.raises(RuntimeError, match="unknown model"):
        fetch_v2v.s2s_specs(settings)


def test_unconfigured_scenarios_yield_no_specs() -> None:
    settings = Settings(coval_s2s_scenarios={"happy-smile": ScenarioCovalIds(test_set_id="TSS")})
    families = {spec.family for spec in fetch_v2v.s2s_specs(settings)}
    assert "s2s-happy-smile" in families
    assert "s2s-bank" not in families and "s2s-happy-customer" not in families


def test_legacy_bank_agent_fields_seed_the_bank_block() -> None:
    settings = Settings(
        coval_s2s_condition_personas={"p": "clean"},
        coval_s2s_bank_test_set_id="TSB",
        coval_s2s_bank_openai_agent_id="b1",
        coval_s2s_bank_stepfun_agent_id=" b5 ",
    )
    assert settings.coval_s2s_scenarios["bank"] == ScenarioCovalIds(
        test_set_id="TSB", agents={"gpt-realtime": "b1", "stepaudio-3-realtime-preview": "b5"}
    )
    explicit = Settings(
        coval_s2s_bank_test_set_id="TSB",
        coval_s2s_bank_openai_agent_id="old",
        coval_s2s_scenarios={
            "bank": ScenarioCovalIds(
                test_set_id="TSB", agents={"gpt-realtime": "new"}, personas={"clean": "p"}
            )
        },
    )
    assert explicit.coval_s2s_scenarios["bank"].agents == {"gpt-realtime": "new"}


def test_a_scenario_with_agents_but_no_personas_is_rejected_at_load() -> None:
    """An empty persona map would classify every caller as clean, so the tiers
    would pool into the clean dataset without a log line."""
    with pytest.raises(ValueError, match=r"\['happy-smile'\] configure agents but no personas"):
        Settings(
            coval_s2s_scenarios={
                "happy-smile": ScenarioCovalIds(test_set_id="TSS", agents={"gpt-realtime": "a1"})
            }
        )


def test_the_legacy_bank_block_may_lean_on_the_legacy_persona_map() -> None:
    settings = Settings(
        coval_s2s_condition_personas={"p": "clean"},
        coval_s2s_bank_test_set_id="TSB",
        coval_s2s_bank_openai_agent_id="b1",
    )
    assert settings.coval_s2s_scenarios["bank"].personas == {}
    with pytest.raises(ValueError, match=r"\['bank'\] configure agents but no personas"):
        Settings(coval_s2s_bank_test_set_id="TSB", coval_s2s_bank_openai_agent_id="b1")


def test_an_unknown_scenario_key_is_rejected_at_load() -> None:
    with pytest.raises(ValueError, match="persona"):
        ScenarioCovalIds.model_validate(
            {"test_set_id": "TSS", "agents": {"gpt-realtime": "a1"}, "persona": {"clean": "p"}}
        )


@pytest.mark.asyncio
async def test_bank_test_set_ids_must_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_bank_test_set_id="TEXT",
        coval_s2s_bank_instruction_metric_id="BIM",
        coval_s2s_scenarios={
            "bank": ScenarioCovalIds(
                test_set_id="VOICE", agents={"gpt-realtime": "b1"}, personas={"clean": "p"}
            )
        },
    )
    monkeypatch.setattr(fetch_v2v, "_client", lambda _s: _fake_client({}, {}))
    with pytest.raises(RuntimeError, match="differs from coval_s2s_bank_test_set_id"):
        await fetch_v2v.fetch_and_write_v2v(settings)


@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_scenario_test_set_id_is_rejected(blank: str) -> None:
    with pytest.raises(ValueError, match="test_set_id"):
        ScenarioCovalIds(test_set_id=blank)
    with pytest.raises(ValueError, match="blank Coval id"):
        ScenarioCovalIds(test_set_id="TS", agents={"gpt-realtime": blank})


def test_scenario_ids_are_stripped() -> None:
    assert ScenarioCovalIds(test_set_id=" TS ").test_set_id == "TS"


@pytest.mark.asyncio
async def test_scenario_personas_do_not_need_the_shared_test_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scenario-only deployment has no coval_s2s_test_set_id; its personas must
    not trip the legacy guard because every spec carries its own test set."""
    settings = Settings(
        coval_s2s_latency_metric_id="MID",
        coval_s2s_bank_instruction_metric_id="BIM",
        coval_s2s_scenarios={
            "bank": ScenarioCovalIds(
                test_set_id="TSB", agents={"gpt-realtime": "b1"}, personas={"clean": "p-clean"}
            )
        },
    )
    assert settings.coval_s2s_test_set_id is None
    writer = _stub_writer()
    client = _fake_client(_list_json(), {})
    fetch_one = AsyncMock(return_value=(RunStatus.SUCCEEDED, 0))

    @contextlib.asynccontextmanager
    async def _fake_pool(_settings: Any) -> AsyncIterator[MagicMock]:
        yield MagicMock()

    monkeypatch.setattr(fetch_v2v, "_client", lambda _s: client)
    monkeypatch.setattr(fetch_v2v, "lifespan_pool", _fake_pool)
    monkeypatch.setattr(fetch_v2v, "RunWriter", lambda _pool: writer)
    monkeypatch.setattr(fetch_v2v, "_fetch_one_provider", fetch_one)

    statuses = await fetch_v2v.fetch_and_write_v2v(settings)

    assert statuses == {"s2s-bank:openai:gpt-realtime": RunStatus.SUCCEEDED}
    fetch_one.assert_awaited_once()
    kwargs = fetch_one.await_args_list[0].kwargs
    assert kwargs["test_set_id"] == "TSB"
    assert kwargs["persona_conditions"] == {"p-clean": Condition.CLEAN}
