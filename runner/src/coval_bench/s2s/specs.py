# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Which Coval agents each benchmark ingests, and the strings their rows carry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from coval_bench import scenarios
from coval_bench.registries import Metric
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import RegisteredModel
from coval_bench.s2s.conditions import (
    ADHERENCE_SLUGS,
    FAMILY_DENTAL,
    FAMILY_MULTITURN,
)

if TYPE_CHECKING:
    from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.s2s.specs")


@dataclass(frozen=True)
class AgentSpec:
    """One provider: its Coval agent id + display strings."""

    provider: str
    model: str
    # None means the agent is not configured and is skipped.
    agent_id: str | None = None
    # Settings attr holding this agent's own Coval test set id; None uses the
    # shared ``coval_s2s_test_set_id``.
    test_set_id_attr: str | None = None
    # The test set id itself, for agents wired through ``coval_s2s_scenarios``;
    # takes precedence over ``test_set_id_attr``.
    test_set_id: str | None = None
    # Dataset family for this agent's rows (see ``s2s.conditions``).
    family: str = FAMILY_MULTITURN
    # Whether this agent's recordings may reach the public samples card.
    publish_samples: bool = True
    benchmark: Benchmark = Benchmark.S2S
    # Settings attr holding the Coval workspace id this agent lives in. None
    # means the workspace coval_api_key defaults to (the existing behavior).
    workspace_id_attr: str | None = None
    # Settings attr holding this agent's own instruction-adherence metric id,
    # overriding the global coval_s2s_instruction_metric_id. None uses the
    # global one.
    instruction_metric_id_attr: str | None = None
    metric_ids: Mapping[Metric, str] = field(default_factory=dict)


# Model strings are display labels. Every agent runs the dental test set: the
# shared multi-turn set stays queryable but is no longer written to, so a spec
# left on it would go stale rather than idle.
def s2s_specs(settings: Settings) -> tuple[AgentSpec, ...]:
    return (
        AgentSpec(
            agent_id=settings.coval_s2s_openai_agent_id,
            provider="openai",
            model="gpt-realtime",
            test_set_id_attr="coval_s2s_dental_test_set_id",
            family=FAMILY_DENTAL,
        ),
        AgentSpec(
            agent_id=settings.coval_s2s_gemini_agent_id,
            provider="google",
            model="gemini-live",
            test_set_id_attr="coval_s2s_dental_test_set_id",
            family=FAMILY_DENTAL,
        ),
        AgentSpec(
            agent_id=settings.coval_s2s_xai_agent_id,
            provider="xai",
            model="grok-voice-think-fast-1.0",
            test_set_id_attr="coval_s2s_dental_test_set_id",
            family=FAMILY_DENTAL,
        ),
        AgentSpec(
            agent_id=settings.coval_s2s_xai_think_fast_2_agent_id,
            provider="xai",
            model="grok-voice-think-fast-2.0",
            test_set_id_attr="coval_s2s_dental_test_set_id",
            family=FAMILY_DENTAL,
        ),
        # Pre-launch models under codenames: the provider and model strings are what
        # land in stored results, so they carry no vendor identity of their own.
        AgentSpec(
            agent_id=settings.coval_s2s_gray_agent_id,
            provider="colors",
            model="gray",
            test_set_id_attr="coval_s2s_dental_test_set_id",
            family=FAMILY_DENTAL,
            publish_samples=False,
        ),
        AgentSpec(
            agent_id=settings.coval_s2s_red_agent_id,
            provider="colors",
            model="red",
            test_set_id_attr="coval_s2s_dental_test_set_id",
            family=FAMILY_DENTAL,
            publish_samples=False,
        ),
        *scenario_specs(settings),
    )


# The models every instruction-following scenario runs, keyed by the model string
# the registry knows them as; the same key names the agent in
# ``Settings.coval_s2s_scenarios``.
SCENARIO_MODELS: tuple[tuple[str, str], ...] = (
    ("openai", "gpt-realtime"),
    ("openai", "gpt-live-1"),
    ("google", "gemini-live"),
    ("google", "gemini-3.8-live"),
    ("google", "gemini-3.8-live-extended-thinking"),
    ("xai", "grok-voice-think-fast-2.0"),
    ("xai", "grok-voice-3.0"),
    ("stepfun", "stepaudio-3-realtime-preview"),
    ("alibaba", "qwen3.8-omni-flash-realtime"),
    ("polyai", "dialog-rsn-1"),
    ("elevenlabs", "eleven_realtime_v1_mini"),
    ("elevenlabs", "eleven_realtime_v1_max"),
    ("rime", "prism"),
    ("phonic", "phonic-1"),
)


def scenario_specs(settings: Settings) -> tuple[AgentSpec, ...]:
    """One spec per (scenario, model); unconfigured ones carry no agent id and are skipped.

    Domain scenarios share the bank Validate Expected Behaviors judge; an
    adherence scenario carries its own judge id and is skipped (with a warning)
    until it names one. Both judges return a fraction rather than a verdict,
    which the instruction mapper scales to a percentage.
    """
    known = {model for _provider, model in SCENARIO_MODELS}
    specs: list[AgentSpec] = []
    for scenario in scenarios.SCENARIOS:
        ids = settings.coval_s2s_scenarios.get(scenario.slug)
        if ids is None:
            continue
        unknown = sorted(set(ids.agents) - known)
        if unknown:
            raise RuntimeError(
                f"coval_s2s_scenarios[{scenario.slug!r}].agents names unknown model(s) "
                f"{unknown}; expected one of {sorted(known)}"
            )
        metric_ids = {
            metric: ident
            for metric, ident in (
                (Metric.INSTRUCTION_FOLLOWING, ids.instruction_metric_id),
                (Metric.WORKFLOW_ADHERENCE, ids.workflow_metric_id),
            )
            if ident
        }
        adherence_metric = ADHERENCE_SLUGS.get(scenario.slug)
        if adherence_metric is not None and adherence_metric not in metric_ids:
            logger.warning(
                "scenario_metric_id_unset", scenario=scenario.slug, metric=adherence_metric.value
            )
            continue
        for provider, model in SCENARIO_MODELS:
            specs.append(
                AgentSpec(
                    agent_id=ids.agents.get(model),
                    provider=provider,
                    model=model,
                    test_set_id=ids.test_set_id,
                    family=scenario.family(Benchmark.S2S),
                    publish_samples=(
                        adherence_metric is None or scenario is scenarios.HEADLINE[Benchmark.S2S]
                    ),
                    instruction_metric_id_attr=(
                        None
                        if adherence_metric is not None
                        else scenarios.ACTIVE.instruction_metric_id_attr
                    ),
                    metric_ids=metric_ids,
                )
            )
    return tuple(specs)


def llm_specs(
    models: Iterable[RegisteredModel], agent_ids: Mapping[str, str] | None = None
) -> tuple[AgentSpec, ...]:
    """Every collected LLM model, driven over the active suite through the proxy.

    TTFT comes from the proxy's own turn log rather than from Coval.
    """
    agent_ids = agent_ids or {}
    return tuple(
        AgentSpec(
            agent_id=agent_ids.get(model.provider),
            provider=model.provider,
            model=model.model,
            test_set_id_attr=scenarios.ACTIVE.test_set_id_attr,
            family=scenarios.ACTIVE.family(Benchmark.LLM),
            publish_samples=False,
            benchmark=Benchmark.LLM,
            instruction_metric_id_attr=scenarios.ACTIVE.instruction_metric_id_attr,
        )
        for model in models
        if model.benchmark is Benchmark.LLM and model.collected
    )
