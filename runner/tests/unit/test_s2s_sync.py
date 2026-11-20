# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from click.testing import CliRunner

from coval_bench.config import Settings
from coval_bench.platform_assets import SyncError
from coval_bench.registries import Metric
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import RegisteredModel
from coval_bench.s2s import sync
from coval_bench.s2s.scenario_map import daily_metrics, daily_scenario
from coval_bench.s2s.sync import SyncRefused, plan_templates, wanted_agents, wanted_metrics

AGENTS = {
    "openai/gpt-realtime-2": ("openai", "gpt-realtime", "A_RT2"),
    "openai/gpt-realtime-2.1": ("openai", "gpt-realtime-2.1", "A_RT21"),
    "google/gemini-3.8-live": ("google", "gemini-3.8-live", "A_GEM"),
}
NAMES = {agent_id: key for key, (_, _, agent_id) in AGENTS.items()}
TEMPLATES = {"clean": "T_CLEAN", "hard": "T_HARD"}
WANTED = {"A_RT21": "openai/gpt-realtime-2.1", "A_GEM": "google/gemini-3.8-live"}
METRIC_IDS = {"judge": "J", "latency": "M_LAT", "call_length": "M_LEN"}
METRIC_NAMES = {coval_id: name for name, coval_id in METRIC_IDS.items()}
ALLOWED: dict[str, tuple[Metric, str]] = {
    "judge": (Metric.WORKFLOW_ADHERENCE, "J"),
    "latency": (Metric.V2V, "M_LAT"),
    "call_length": (Metric.CALL_LENGTH, "M_LEN"),
}
ALL_COLLECTED = frozenset({"WorkflowAdherence", "V2V", "CallLength"})


class _FakeClient:
    def __init__(
        self,
        live: dict[str, Any],
        fail_on_update: str | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        self.live = live
        self.fail_on_update = fail_on_update
        self.metrics = sorted(METRIC_NAMES) if metrics is None else metrics
        self.updates: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_run_template(self, template_id: str) -> dict[str, object]:
        return {
            "id": template_id,
            "agent_ids": self.live[template_id],
            "metric_ids": self.metrics,
        }

    def update_run_template(self, template_id: str, body: dict[str, object]) -> dict[str, object]:
        if template_id == self.fail_on_update:
            raise SyncError(f"PATCH /run-templates/{template_id} -> 503: <html>secret body</html>")
        self.updates.append((template_id, body))
        return {"id": template_id, **body}


def test_wanted_agents_keeps_only_collected_rows() -> None:
    collected = {("openai", "gpt-realtime-2.1"), ("google", "gemini-3.8-live")}

    assert wanted_agents(AGENTS, collected) == WANTED


def test_wanted_agents_refuses_to_empty_the_templates() -> None:
    with pytest.raises(SyncRefused, match="not emptying"):
        wanted_agents(AGENTS, set())


def test_wanted_agents_skips_an_agent_without_a_provider() -> None:
    agents = {**AGENTS, "odd": (None, "gemini-3.8-live", "A_ODD")}

    assert "A_ODD" not in wanted_agents(agents, {("google", "gemini-3.8-live")})


def test_plan_swaps_a_retired_agent_for_an_enrolled_one() -> None:
    client = _FakeClient({"T_CLEAN": ["A_RT2", "A_GEM"], "T_HARD": ["A_RT2", "A_GEM"]})

    changes = plan_templates(client, TEMPLATES, WANTED, NAMES)

    assert [(c.condition, c.added, c.removed) for c in changes] == [
        ("clean", ("openai/gpt-realtime-2.1",), ("openai/gpt-realtime-2",)),
        ("hard", ("openai/gpt-realtime-2.1",), ("openai/gpt-realtime-2",)),
    ]
    assert changes[0].agent_ids == ("A_GEM", "A_RT21")
    assert client.updates == []


def test_plan_leaves_a_matching_template_alone() -> None:
    client = _FakeClient({"T_CLEAN": ["A_GEM", "A_RT21"], "T_HARD": ["A_GEM", "A_RT21"]})

    assert not any(c.changed for c in plan_templates(client, TEMPLATES, WANTED, NAMES))


def test_plan_keeps_an_agent_the_map_does_not_define() -> None:
    client = _FakeClient({"T_CLEAN": ["A_GEM", "A_RT21", "A_STRAY"], "T_HARD": ["A_GEM", "A_RT21"]})

    clean, hard = plan_templates(client, TEMPLATES, WANTED, NAMES)

    assert clean.unknown == ("A_STRAY",)
    assert not clean.changed
    assert "A_STRAY" in clean.agent_ids
    assert hard.unknown == ()


def test_plan_refuses_a_template_without_an_agent_list() -> None:
    client = _FakeClient({"T_CLEAN": None, "T_HARD": ["A_GEM"]})

    with pytest.raises(SyncRefused, match="no agent_ids list"):
        plan_templates(client, TEMPLATES, WANTED, NAMES)


def test_plan_refuses_a_scenario_without_templates() -> None:
    with pytest.raises(SyncRefused, match="no run templates"):
        plan_templates(_FakeClient({}), {}, WANTED, NAMES)


def test_wanted_metrics_keeps_only_collected_ones() -> None:
    wanted = wanted_metrics(ALLOWED, {"WorkflowAdherence", "CallLength"}, Metric.WORKFLOW_ADHERENCE)

    assert wanted == {"J": "judge", "M_LEN": "call_length"}


def test_wanted_metrics_refuses_to_drop_the_anchor() -> None:
    with pytest.raises(SyncRefused, match="anchors"):
        wanted_metrics(ALLOWED, {"V2V", "CallLength"}, Metric.WORKFLOW_ADHERENCE)
    with pytest.raises(SyncRefused, match="does not list"):
        wanted_metrics({"latency": ALLOWED["latency"]}, ALL_COLLECTED, Metric.WORKFLOW_ADHERENCE)


def test_plan_drops_an_uncollected_metric_and_keeps_a_stray_one() -> None:
    live = {"T_CLEAN": ["A_GEM", "A_RT21"], "T_HARD": ["A_GEM", "A_RT21"]}
    client = _FakeClient(live, metrics=["J", "M_LAT", "M_HAND"])
    metrics = {"J": "judge", "M_LEN": "call_length"}

    changes = plan_templates(client, TEMPLATES, WANTED, NAMES, metrics, METRIC_NAMES)

    assert [(c.metrics_added, c.metrics_removed, c.unknown_metrics) for c in changes] == [
        (("call_length",), ("latency",), ("M_HAND",))
    ] * 2
    assert changes[0].body == {"metric_ids": ["J", "M_HAND", "M_LEN"]}


def test_plan_leaves_metrics_alone_without_flags() -> None:
    client = _FakeClient(
        {"T_CLEAN": ["A_GEM", "A_RT21"], "T_HARD": ["A_GEM", "A_RT21"]}, metrics=[]
    )

    assert not any(c.changed for c in plan_templates(client, TEMPLATES, WANTED, NAMES))


def test_daily_metrics_names_the_judge_and_the_built_ins() -> None:
    version = _version(enabled=True)
    version["metric_ids"] = {**METRIC_IDS, "talk_ratio": "M_TALK"}

    assert daily_metrics(daily_scenario({"workflow-bank": {"v1": version}})) == ALLOWED


def _version(*, enabled: bool) -> dict[str, object]:
    return {
        "test_set_id": "TSW",
        "agents": {
            "openai/gpt-realtime-2": {
                "id": "A_RT2",
                "registry_model": "gpt-realtime",
                "provider": "openai",
            },
            "openai/gpt-realtime-2.1": {
                "id": "A_RT21",
                "registry_model": "gpt-realtime-2.1",
                "provider": "openai",
            },
            "google/gemini-3.8-live": {
                "id": "A_GEM",
                "registry_model": "gemini-3.8-live",
                "provider": "google",
            },
        },
        "personas": {"clean": {"id": "P"}},
        "judge": {"id": "J"},
        "metric_ids": METRIC_IDS,
        "run_template_ids": TEMPLATES,
        "schedules": {"clean": {"enabled": enabled}},
    }


def test_daily_scenario_reads_the_live_workflow_bank_templates() -> None:
    raw = {"workflow-bank": {"v1": _version(enabled=True), "v2": _version(enabled=False)}}

    scenario = daily_scenario(raw)

    assert scenario.run_template_ids == TEMPLATES
    assert scenario.agents["google/gemini-3.8-live"].id == "A_GEM"


def test_daily_scenario_needs_an_enabled_schedule() -> None:
    with pytest.raises(ValueError, match="enabled schedule"):
        daily_scenario({"workflow-bank": {"v1": _version(enabled=False)}})


def _row(provider: str, model: str, *, collected: bool) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.S2S,
        provider=provider,
        model=model,
        collected=collected,
        visibility="public",
    )


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeClient,
    *args: str,
    collected: frozenset[str] = ALL_COLLECTED,
) -> tuple[int, list[dict[str, object]]]:
    published: list[dict[str, object]] = []
    settings = Settings.model_construct(
        log_level="INFO", s2s_scenario_map_uri="gs://b/k", alerts_topic="projects/p/topics/alerts"
    )
    monkeypatch.setattr(sync, "get_settings", lambda: settings)
    monkeypatch.setattr(sync, "configure_logging", lambda level: None)
    monkeypatch.setattr(
        sync, "read_scenario_map", lambda uri: {"workflow-bank": {"v1": _version(enabled=True)}}
    )
    monkeypatch.setattr(
        sync,
        "load_models",
        lambda _s: [
            _row("openai", "gpt-realtime", collected=False),
            _row("openai", "gpt-realtime-2.1", collected=True),
            _row("google", "gemini-3.8-live", collected=True),
        ],
    )
    monkeypatch.setattr(sync, "load_collected_metrics", lambda _s: collected)
    monkeypatch.setenv("COVAL_API_KEY", "k")
    monkeypatch.setattr(sync, "CovalClient", lambda _key, _base: client)
    monkeypatch.setattr(sync, "publish_coval_failure", lambda topic, **card: published.append(card))
    result = CliRunner().invoke(sync.sync_s2s, list(args))
    return result.exit_code, published


def test_cli_updates_every_changed_template(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient({"T_CLEAN": ["A_RT2", "A_GEM"], "T_HARD": ["A_RT2", "A_GEM"]})

    code, published = _run_cli(monkeypatch, client)

    assert code == 0
    assert client.updates == [
        ("T_CLEAN", {"agent_ids": ["A_GEM", "A_RT21"]}),
        ("T_HARD", {"agent_ids": ["A_GEM", "A_RT21"]}),
    ]
    assert published == []


def test_cli_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient({"T_CLEAN": ["A_RT2"], "T_HARD": ["A_RT2"]})

    code, _ = _run_cli(monkeypatch, client, "--dry-run")

    assert code == 0
    assert client.updates == []


def test_cli_coval_failure_posts_a_card_without_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(
        {"T_CLEAN": ["A_RT2", "A_GEM"], "T_HARD": ["A_RT2", "A_GEM"]}, fail_on_update="T_HARD"
    )

    code, published = _run_cli(monkeypatch, client)

    assert code != 0
    assert client.updates == [("T_CLEAN", {"agent_ids": ["A_GEM", "A_RT21"]})]
    (card,) = published
    assert card["job"] == "s2s-sync"
    assert card["error"] == "PATCH /run-templates/T_HARD -> 503; updated before failing: clean"
    assert "secret body" not in str(card)


def test_cli_drops_a_metric_that_is_not_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient({"T_CLEAN": ["A_GEM", "A_RT21"], "T_HARD": ["A_GEM", "A_RT21"]})

    code, _ = _run_cli(
        monkeypatch, client, collected=frozenset({"WorkflowAdherence", "CallLength"})
    )

    assert code == 0
    assert client.updates == [
        ("T_CLEAN", {"metric_ids": ["J", "M_LEN"]}),
        ("T_HARD", {"metric_ids": ["J", "M_LEN"]}),
    ]


def test_cli_syncs_agents_and_alerts_when_the_anchor_is_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient({"T_CLEAN": ["A_RT2", "A_GEM"], "T_HARD": ["A_GEM", "A_RT21"]})

    code, published = _run_cli(monkeypatch, client, collected=frozenset({"V2V"}))

    assert code == 0
    assert client.updates == [("T_CLEAN", {"agent_ids": ["A_GEM", "A_RT21"]})]
    (card,) = published
    assert (card["failure"], "anchors" in str(card["error"])) == ("metric_sync", True)


def test_cli_still_syncs_agents_when_the_flags_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient({"T_CLEAN": ["A_RT2", "A_GEM"], "T_HARD": ["A_GEM", "A_RT21"]}, metrics=[])

    def unreadable(_settings: Settings) -> frozenset[str]:
        raise psycopg.errors.UndefinedColumn("collected")

    code, _ = _run_cli(monkeypatch, client)
    assert code == 0
    client.updates.clear()
    monkeypatch.setattr(sync, "load_collected_metrics", unreadable)
    result = CliRunner().invoke(sync.sync_s2s, [])

    assert result.exit_code == 0
    assert client.updates == [("T_CLEAN", {"agent_ids": ["A_GEM", "A_RT21"]})]
