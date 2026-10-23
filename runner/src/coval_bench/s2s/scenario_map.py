# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The S2S fetch's Coval ids, read from the scenario map each assets apply publishes.

Reading it when the fetch runs means a new agent needs only the assets apply, not a
re-plan of the prod job that used to copy these ids into its environment.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from coval_bench import gcs
from coval_bench.config import ScenarioCovalIds, Settings

FETCHED_SCENARIOS: tuple[str, ...] = ("bank", "workflow-bank")
_JUDGE_FIELDS = {"workflow-bank": "workflow_metric_id"}
_BANK_METRICS = {
    "coval_s2s_latency_metric_id": "latency",
    "coval_s2s_interruption_metric_id": "interruption_rate",
    "coval_s2s_call_length_metric_id": "call_length",
}
_CLEAN_PERSONA_LABEL = "Clean Speech Baseline"


class _Ref(BaseModel, frozen=True):
    id: str


class _Agent(BaseModel, frozen=True):
    id: str
    registry_model: str


class _Schedule(BaseModel, frozen=True):
    enabled: bool


class _ScenarioVersion(BaseModel, frozen=True):
    test_set_id: str
    agents: dict[str, _Agent]
    personas: dict[str, _Ref]
    judge: _Ref
    metric_ids: dict[str, str]
    schedules: dict[str, _Schedule]


def _live_version(name: str, versions: object) -> _ScenarioVersion:
    """The scenario's newest version with an enabled schedule."""
    if not isinstance(versions, dict):
        raise ValueError(f"scenario map has no {name!r} scenario")
    live: list[tuple[int, _ScenarioVersion]] = []
    for version, assets in versions.items():
        parsed = _ScenarioVersion.model_validate(assets)
        if any(schedule.enabled for schedule in parsed.schedules.values()):
            live.append((int(version.removeprefix("v")), parsed))
    if not live:
        raise ValueError(f"scenario map has no {name!r} version with an enabled schedule")
    return max(live, key=lambda pair: pair[0])[1]


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
    judge = {field: assets.judge.id for field in [_JUDGE_FIELDS.get(name)] if field}
    return ScenarioCovalIds(
        test_set_id=assets.test_set_id,
        agents=agents,
        personas={condition: persona.id for condition, persona in assets.personas.items()},
        **judge,
    )


def scenario_map_fields(raw: object) -> dict[str, Any]:
    """The Settings fields the fetch reads, built from a parsed scenario map."""
    if not isinstance(raw, dict):
        raise ValueError("scenario map is not a JSON object")
    live = {name: _live_version(name, raw.get(name)) for name in FETCHED_SCENARIOS}
    scenarios = {name: _scenario_ids(name, assets) for name, assets in live.items()}
    bank = live["bank"]
    missing = sorted(key for key in _BANK_METRICS.values() if not bank.metric_ids.get(key))
    if missing:
        raise ValueError(f"scenario map 'bank' has no metric id for {missing}")
    if not bank.judge.id.strip():
        raise ValueError("scenario map 'bank' has no judge id")
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
