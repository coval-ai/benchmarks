# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The S2S fetch's Coval ids, read from the scenario map each assets apply publishes.

Reading it when the fetch runs means a new agent needs only the assets apply, not a
re-plan of the prod job that used to copy these ids into its environment.
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from pydantic import BaseModel, StringConstraints

from coval_bench import gcs
from coval_bench.config import ScenarioCovalIds, Settings
from coval_bench.registries import Metric
from coval_bench.s2s.conditions import PLATFORM_SLUGS

logger = structlog.get_logger("coval_bench.s2s.scenario_map")

FETCHED_SCENARIOS: tuple[str, ...] = ("bank", "workflow-bank")
_JUDGE_FIELDS = {"workflow-bank": "workflow_metric_id"}
_BANK_METRICS = {
    "coval_s2s_latency_metric_id": "latency",
    "coval_s2s_interruption_metric_id": "interruption_rate",
    "coval_s2s_call_length_metric_id": "call_length",
}
_CLEAN_PERSONA_LABEL = "Clean Speech Baseline"
# Metrics the platform rows can carry; any other map key (e.g. one Coval adds before
# we store it) is left out rather than failing the fetch.
_PLATFORM_METRICS = {
    "latency": Metric.V2V,
    "interruption_rate": Metric.INTERRUPTION_RATE,
    "call_length": Metric.CALL_LENGTH,
    "character_error_rate": Metric.CHARACTER_ERROR_RATE,
    "tool_call_count": Metric.TOOL_CALL_COUNT,
}

# Coval ids go into run filters as agent_id="…", so quotes or spaces must never pass.
_CovalId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$")]


class _Ref(BaseModel, frozen=True):
    id: _CovalId


class _Agent(BaseModel, frozen=True):
    id: _CovalId
    registry_model: str
    provider: str | None = None
    variant_id: str | None = None


class _Schedule(BaseModel, frozen=True):
    enabled: bool


class _Schedules(BaseModel, frozen=True):
    schedules: dict[str, _Schedule]


class _ScenarioVersion(_Schedules, frozen=True):
    test_set_id: _CovalId
    agents: dict[str, _Agent]
    personas: dict[str, _Ref]
    judge: _Ref | None
    metric_ids: dict[str, _CovalId]


def _newest_live(versions: dict[str, Any]) -> _ScenarioVersion | None:
    """The newest version with an enabled schedule, or None when none is scheduled."""
    live = [
        (int(version.removeprefix("v")), assets)
        for version, assets in versions.items()
        if any(s.enabled for s in _Schedules.model_validate(assets).schedules.values())
    ]
    if not live:
        return None
    return _ScenarioVersion.model_validate(max(live, key=lambda pair: pair[0])[1])


def _live_version(name: str, versions: object) -> _ScenarioVersion:
    """The scenario's newest version with an enabled schedule."""
    if not isinstance(versions, dict):
        raise ValueError(f"scenario map has no {name!r} scenario")
    live = _newest_live(versions)
    if live is None:
        raise ValueError(f"scenario map has no {name!r} version with an enabled schedule")
    return live


def _scenario_ids(name: str, assets: _ScenarioVersion) -> ScenarioCovalIds:
    if not assets.agents:
        raise ValueError(f"scenario map {name!r} has no agents")
    if "clean" not in assets.personas:
        raise ValueError(f"scenario map {name!r} has no clean persona")
    agents: dict[str, str] = {}
    for agent in assets.agents.values():
        if agent.registry_model in agents:
            raise ValueError(f"scenario map {name!r} repeats model {agent.registry_model!r}")
        agents[agent.registry_model] = agent.id
    field = _JUDGE_FIELDS.get(name)
    if field and assets.judge is None:
        raise ValueError(f"scenario map {name!r} has no judge")
    judge = {field: assets.judge.id} if field and assets.judge else {}
    return ScenarioCovalIds(
        test_set_id=assets.test_set_id,
        agents=agents,
        personas={condition: persona.id for condition, persona in assets.personas.items()},
        **judge,
    )


def _platform_ids(name: str, assets: _ScenarioVersion) -> ScenarioCovalIds:
    """A platform scenario's ids, its agents keyed "provider/model" like registry rows."""
    agents: dict[str, str] = {}
    variant_ids: dict[str, str] = {}
    for agent in assets.agents.values():
        if not agent.provider:
            raise ValueError(f"scenario map {name!r} has an agent without a provider")
        key = f"{agent.provider}/{agent.registry_model}"
        if key in agents:
            raise ValueError(f"scenario map {name!r} repeats agent {key!r}")
        agents[key] = agent.id
        if agent.variant_id:
            variant_ids[key] = agent.variant_id
    return ScenarioCovalIds(
        test_set_id=assets.test_set_id,
        agents=agents,
        personas={condition: persona.id for condition, persona in assets.personas.items()},
        variant_ids=variant_ids,
        metric_ids={
            metric.value: assets.metric_ids[key]
            for key, metric in _PLATFORM_METRICS.items()
            if assets.metric_ids.get(key)
        },
    )


def scenario_map_fields(raw: object) -> dict[str, Any]:
    """The Settings fields the fetch reads, built from a parsed scenario map."""
    if not isinstance(raw, dict):
        raise ValueError("scenario map is not a JSON object")
    live = {name: _live_version(name, raw.get(name)) for name in FETCHED_SCENARIOS}
    scenarios = {name: _scenario_ids(name, assets) for name, assets in live.items()}
    for name in PLATFORM_SLUGS:
        versions = raw.get(name)
        assets = _newest_live(versions) if isinstance(versions, dict) else None
        if assets is None:
            logger.info("platform_scenario_not_scheduled", scenario=name)
            continue
        scenarios[name] = _platform_ids(name, assets)
    bank = live["bank"]
    if bank.judge is None:
        raise ValueError("scenario map 'bank' has no judge")
    missing = sorted(key for key in _BANK_METRICS.values() if not bank.metric_ids.get(key))
    if missing:
        raise ValueError(f"scenario map 'bank' has no metric id for {missing}")
    clean_persona_id = bank.personas["clean"].id
    return {
        "coval_s2s_scenarios": scenarios,
        **{field: bank.metric_ids[key] for field, key in _BANK_METRICS.items()},
        "coval_s2s_bank_test_set_id": bank.test_set_id,
        "coval_s2s_bank_instruction_metric_id": bank.judge.id,
        "coval_s2s_bank_persona_id": clean_persona_id,
        "s2s_persona_labels": {clean_persona_id: _CLEAN_PERSONA_LABEL},
    }


def with_scenario_map(settings: Settings) -> Settings:
    """``settings`` with its Coval ids taken from the scenario map, when one is configured."""
    uri = settings.s2s_scenario_map_uri.strip()
    if not uri:
        return settings
    bucket, _, key = uri.removeprefix("gs://").partition("/")
    if not uri.startswith("gs://") or not bucket or not key:
        raise ValueError(f"s2s_scenario_map_uri {uri!r} is not a gs://bucket/key URI")
    raw = gcs.read_json(bucket, key)
    if raw is None:
        raise RuntimeError(f"scenario map {uri} not found")
    return Settings.model_validate({**settings.model_dump(), **scenario_map_fields(raw)})
