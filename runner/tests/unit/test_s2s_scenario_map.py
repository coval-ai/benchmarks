# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reading the S2S fetch's Coval ids from the scenario map."""

from __future__ import annotations

from typing import Any

import pytest

from coval_bench.config import ScenarioCovalIds, Settings
from coval_bench.s2s.scenario_map import scenario_map_fields, with_scenario_map

_URI = "gs://coval-bench-asset-versions/coval-benchmarking/scenario_map.json"


def _version(
    *,
    agents: dict[str, str],
    enabled: bool = True,
    test_set_id: str = "TS",
    judge: str = "JUDGE",
    personas: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "workspace_id": "WS",
        "test_set_id": test_set_id,
        "agents": {
            model: {"id": ident, "provider": "p", "model": model, "registry_model": model}
            for model, ident in agents.items()
        },
        "personas": {
            condition: {"id": ident, "version": "v1"}
            for condition, ident in (personas or {"clean": "P0", "hard": "P3"}).items()
        },
        "judge": {"id": judge, "name": "j", "version": "v1", "version_number": 1},
        "metric_ids": {
            "judge": judge,
            "latency": "LAT",
            "interruption_rate": "INT",
            "call_length": "LEN",
        },
        "schedules": {
            "clean": {
                "id": "S",
                "enabled": enabled,
                "schedule_expression": "cron(0 17 * * ? *)",
                "schedule_timezone": "America/Los_Angeles",
            }
        },
    }


def _map(**overrides: Any) -> dict[str, Any]:
    scenario_map: dict[str, Any] = {
        "bank": {"v1": _version(agents={"gpt-realtime": "A1", "gemini-live": "A2"})},
        "workflow-bank": {
            "v1": _version(agents={"gpt-realtime": "A1"}, test_set_id="TSW", judge="WJ")
        },
        "dental": {"v1": {"unrelated": True}},
    }
    scenario_map.update(overrides)
    return scenario_map


def test_fields_match_what_terraform_set() -> None:
    fields = scenario_map_fields(_map())

    assert fields["coval_s2s_scenarios"] == {
        "bank": ScenarioCovalIds(
            test_set_id="TS",
            agents={"gpt-realtime": "A1", "gemini-live": "A2"},
            personas={"clean": "P0", "hard": "P3"},
        ),
        "workflow-bank": ScenarioCovalIds(
            test_set_id="TSW",
            agents={"gpt-realtime": "A1"},
            personas={"clean": "P0", "hard": "P3"},
            workflow_metric_id="WJ",
        ),
    }
    assert fields["coval_s2s_latency_metric_id"] == "LAT"
    assert fields["coval_s2s_interruption_metric_id"] == "INT"
    assert fields["coval_s2s_call_length_metric_id"] == "LEN"
    assert fields["coval_s2s_bank_test_set_id"] == "TS"
    assert fields["coval_s2s_bank_instruction_metric_id"] == "JUDGE"
    assert fields["coval_s2s_bank_persona_id"] == "P0"
    assert fields["s2s_persona_labels"] == {"P0": "Clean Speech Baseline"}


def test_newest_version_with_an_enabled_schedule_wins() -> None:
    bank = {
        "v1": _version(agents={"gpt-realtime": "OLD"}),
        "v2": _version(agents={"gpt-realtime": "NEW"}),
        "v3": _version(agents={"gpt-realtime": "PAUSED"}, enabled=False),
    }

    fields = scenario_map_fields(_map(bank=bank))

    assert fields["coval_s2s_scenarios"]["bank"].agents == {"gpt-realtime": "NEW"}


def test_scenario_without_an_enabled_schedule_is_rejected() -> None:
    bank = {"v1": _version(agents={"gpt-realtime": "A1"}, enabled=False)}

    with pytest.raises(ValueError, match="no 'bank' version with an enabled schedule"):
        scenario_map_fields(_map(bank=bank))


def test_missing_scenario_is_rejected() -> None:
    scenario_map = _map()
    del scenario_map["workflow-bank"]

    with pytest.raises(ValueError, match="no 'workflow-bank' scenario"):
        scenario_map_fields(scenario_map)


def test_repeated_registry_model_is_rejected() -> None:
    version = _version(agents={"a": "A1", "b": "A2"})
    version["agents"]["b"]["registry_model"] = "a"

    with pytest.raises(ValueError, match="repeats model 'a'"):
        scenario_map_fields(_map(bank={"v1": version}))


def test_scenario_without_a_clean_persona_is_rejected() -> None:
    bank = {"v1": _version(agents={"gpt-realtime": "A1"}, personas={"hard": "P3"})}

    with pytest.raises(ValueError, match="'bank' has no clean persona"):
        scenario_map_fields(_map(bank=bank))


def test_missing_bank_metric_is_rejected() -> None:
    version = _version(agents={"gpt-realtime": "A1"})
    del version["metric_ids"]["call_length"]

    with pytest.raises(ValueError, match=r"no metric id for \['call_length'\]"):
        scenario_map_fields(_map(bank={"v1": version}))


def test_without_a_uri_settings_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("read the scenario map without a URI")

    monkeypatch.setattr("coval_bench.gcs.read_json", _explode)
    settings = Settings(_env_file=None)

    assert with_scenario_map(settings) is settings


def test_map_replaces_the_env_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[tuple[str, str]] = []

    def _read_json(bucket: str, key: str) -> object:
        reads.append((bucket, key))
        return _map()

    monkeypatch.setattr("coval_bench.gcs.read_json", _read_json)
    settings = Settings(
        _env_file=None,
        s2s_scenario_map_uri=_URI,
        coval_s2s_latency_metric_id="STALE",
        log_level="DEBUG",
    )

    loaded = with_scenario_map(settings)

    assert reads == [("coval-bench-asset-versions", "coval-benchmarking/scenario_map.json")]
    assert loaded.coval_s2s_latency_metric_id == "LAT"
    assert loaded.coval_s2s_scenarios["bank"].agents == {"gpt-realtime": "A1", "gemini-live": "A2"}
    assert loaded.log_level == "DEBUG"


def test_missing_map_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("coval_bench.gcs.read_json", lambda bucket, key: None)

    with pytest.raises(RuntimeError, match="not found"):
        with_scenario_map(Settings(_env_file=None, s2s_scenario_map_uri=_URI))


def test_uri_must_name_a_gcs_object() -> None:
    with pytest.raises(ValueError, match="not a gs://bucket/key URI"):
        with_scenario_map(Settings(_env_file=None, s2s_scenario_map_uri="gs://bucket-only"))


def test_an_old_version_of_another_shape_is_ignored() -> None:
    bank = {
        "v1": {"schedules": {"clean": {"enabled": False}}, "agents": "pre-registry"},
        "v2": _version(agents={"gpt-realtime": "A1"}),
    }

    fields = scenario_map_fields(_map(bank=bank))

    assert fields["coval_s2s_scenarios"]["bank"].agents == {"gpt-realtime": "A1"}


@pytest.mark.parametrize("bad", ['A1" OR agent_id="X', "A 1", ""])
def test_ids_outside_coval_format_are_rejected(bad: str) -> None:
    bank = {"v1": _version(agents={"gpt-realtime": bad})}

    with pytest.raises(ValueError, match="pattern"):
        scenario_map_fields(_map(bank=bank))
