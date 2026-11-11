# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fetching the interruption suites Corral runs on its own agents."""

from __future__ import annotations

import httpx
import pytest

from coval_bench.config import ScenarioCovalIds, Settings
from coval_bench.registries import Metric
from coval_bench.s2s import ingest, specs
from coval_bench.s2s.conditions import Condition, DatasetMetrics, condition_for, dataset_id_for
from coval_bench.s2s.coval_runs import CovalRun
from coval_bench.s2s.specs import AgentSpec

YIELD = ScenarioCovalIds(
    test_set_id="TSY",
    personas={"clean": "CUE"},
    metric_ids={"InterruptionCueLanded": "GATE", "InterruptionYielded": "YIELD"},
    workspace_id="WS",
)


def _client(agents: dict[str, str], seen: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        wanted = request.url.params["filter"].removeprefix('customer_agent_id="').rstrip('"')
        found = [{"id": agents[wanted], "customer_agent_id": wanted}] if wanted in agents else []
        return httpx.Response(200, json={"agents": found})

    return httpx.AsyncClient(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))


def test_customer_agent_id_matches_corral() -> None:
    assert (
        specs.corral_customer_agent_id("google", "gemini-3.8-live", "WS")
        == "google-gemini-3.8-live--sdk--WS"
    )
    assert (
        specs.corral_customer_agent_id("elevenlabs", "eleven_realtime_v1_mini", "WS")
        == "elevenlabs-eleven-realtime-v1-mini--sdk--WS"
    )


@pytest.mark.asyncio
async def test_harness_specs_keep_only_models_corral_has_an_agent_for() -> None:
    settings = Settings(coval_s2s_scenarios={"interruption-yield": YIELD})
    seen: list[httpx.Request] = []
    agents = {"google-gemini-3.8-live--sdk--WS": "AG"}

    async with _client(agents, seen) as client:
        found = await specs.harness_specs(
            client, settings, [("google", "gemini-3.8-live"), ("rime", "prism")]
        )

    assert found == (
        AgentSpec(
            agent_id="AG",
            provider="google",
            model="gemini-3.8-live",
            test_set_id="TSY",
            family="s2s-interruption-yield",
            publish_samples=False,
            metric_ids={
                Metric.INTERRUPTION_CUE_LANDED: "GATE",
                Metric.INTERRUPTION_YIELDED: "YIELD",
            },
            uses_shared_metrics=False,
            workspace_id="WS",
        ),
    )
    assert {r.headers["X-Coval-Workspace-Id"] for r in seen} == {"WS"}


@pytest.mark.asyncio
async def test_harness_specs_look_each_agent_up_once_across_suites() -> None:
    backchannel = YIELD.model_copy(
        update={
            "test_set_id": "TSB",
            "metric_ids": {"InterruptionCueLanded": "GATE", "BackchannelContinued": "CONT"},
        }
    )
    settings = Settings(
        coval_s2s_scenarios={"interruption-yield": YIELD, "interruption-backchannel": backchannel}
    )
    seen: list[httpx.Request] = []

    async with _client({"xai-grok-voice-3.0--sdk--WS": "AX"}, seen) as client:
        found = await specs.harness_specs(client, settings, [("xai", "grok-voice-3.0")])

    assert [(s.family, s.test_set_id) for s in found] == [
        ("s2s-interruption-yield", "TSY"),
        ("s2s-interruption-backchannel", "TSB"),
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_no_harness_suite_means_no_lookups() -> None:
    seen: list[httpx.Request] = []
    async with _client({}, seen) as client:
        assert await specs.harness_specs(client, Settings(), [("xai", "grok-voice-3.0")]) == ()
    assert seen == []


@pytest.mark.parametrize(
    ("slug", "judge"),
    [
        ("interruption-yield", Metric.INTERRUPTION_YIELDED),
        ("interruption-backchannel", Metric.BACKCHANNEL_CONTINUED),
    ],
)
def test_each_suite_is_one_clean_dataset_anchored_on_the_cue(slug: str, judge: Metric) -> None:
    family = f"s2s-{slug}"
    dataset_id = dataset_id_for(family, Condition.CLEAN)

    assert dataset_id == f"{family}-v1"
    assert dataset_id_for(family, Condition.HARD) is None
    contract = condition_for(dataset_id)
    assert contract.required is Metric.INTERRUPTION_CUE_LANDED
    assert contract.optional == {judge}


@pytest.mark.asyncio
async def test_a_failed_agent_lookup_skips_that_model_without_stopping_the_fetch() -> None:
    settings = Settings(coval_s2s_scenarios={"interruption-yield": YIELD})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://api.test/v1", transport=transport) as client:
        assert await specs.harness_specs(client, settings, [("google", "gemini-3.8-live")]) == ()


def _judged(cue_values: list[object], *, only_where_anchor_positive: bool = True) -> list[str]:
    anchor = [
        {"simulation_output_id": f"s{i}", "value": value} for i, value in enumerate(cue_values)
    ]
    verdicts = [{"simulation_output_id": f"s{i}", "value": "YES"} for i in range(len(cue_values))]
    writable = ingest._writable_metrics(
        spec=AgentSpec(provider="google", model="gemini-live"),
        coval_run=CovalRun(run_id="RUN", create_time=None),
        metrics={"YIELD": {"values": verdicts}},
        anchor=Metric.INTERRUPTION_CUE_LANDED,
        anchor_values=anchor,
        condition=DatasetMetrics(
            required=Metric.INTERRUPTION_CUE_LANDED,
            optional=frozenset({Metric.INTERRUPTION_YIELDED}),
            optional_only_where_anchor_positive=only_where_anchor_positive,
        ),
        pending=frozenset({Metric.INTERRUPTION_CUE_LANDED, Metric.INTERRUPTION_YIELDED}),
        metric_ids={Metric.INTERRUPTION_YIELDED: "YIELD"},
    )
    assert len(writable[Metric.INTERRUPTION_CUE_LANDED]) == len(cue_values)
    return [v["simulation_output_id"] for v in writable.get(Metric.INTERRUPTION_YIELDED, [])]


def test_verdicts_count_only_where_the_cue_landed() -> None:
    assert _judged([2, 1.0, 0, None, "x", True]) == ["s0", "s1"]


def test_no_verdict_is_written_when_no_cue_landed() -> None:
    assert _judged([0, 0]) == []


def test_other_datasets_keep_every_verdict() -> None:
    assert _judged([2, 0, None], only_where_anchor_positive=False) == ["s0", "s1", "s2"]


@pytest.mark.parametrize("slug", ["interruption-yield", "interruption-backchannel"])
def test_both_suites_filter_on_the_cue(slug: str) -> None:
    dataset_id = dataset_id_for(f"s2s-{slug}", Condition.CLEAN)
    assert dataset_id is not None
    assert condition_for(dataset_id).optional_only_where_anchor_positive
