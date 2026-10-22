# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that the S2S fetch settings find recent Coval runs, without writing anything."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import click
import httpx

from coval_bench.config import Settings
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.coval_runs import coval_client, recent_completed_runs
from coval_bench.s2s.fetch_v2v import validate_fetch_settings
from coval_bench.s2s.identity import (
    UnmappedPersona,
    all_persona_conditions,
    condition_for_persona,
    parse_persona_conditions,
)
from coval_bench.s2s.specs import s2s_specs, scenario_specs


@dataclass(frozen=True)
class ModelRuns:
    scenario: str
    model: str
    runs_by_condition: dict[str, int]


async def check_scenarios(
    settings: Settings, client: httpx.AsyncClient, *, window_seconds: int | None = None
) -> tuple[list[ModelRuns], list[str]]:
    """Count each scenario model's recent runs per condition and flag what the fetch would miss."""
    validate_fetch_settings(settings, s2s_specs(settings), Benchmark.S2S)
    persona_conditions = parse_persona_conditions(all_persona_conditions(settings))
    found: list[ModelRuns] = []
    flags: list[str] = []
    unmapped: dict[str, set[str]] = {}
    for slug, ids in settings.coval_s2s_scenarios.items():
        only_this = settings.model_copy(update={"coval_s2s_scenarios": {slug: ids}})
        specs = [s for s in scenario_specs(only_this) if s.agent_id]
        if not specs:
            flags.append(f"{slug}: the fetch skips this scenario (no agents or no judge)")
            continue
        for spec in specs:
            if spec.agent_id is None:
                continue
            attr = spec.instruction_metric_id_attr
            if attr and not (getattr(settings, attr) or "").strip():
                flags.append(f"{slug}/{spec.model}: the fetch skips it, {attr} is unset")
                continue
            runs = await recent_completed_runs(
                client,
                spec.agent_id,
                period_seconds=settings.s2s_fetch_period_seconds,
                test_set_id=spec.test_set_id,
                window_seconds=window_seconds,
            )
            counts = dict.fromkeys(ids.personas, 0)
            for run in runs:
                try:
                    condition = condition_for_persona(
                        run.persona_id, persona_conditions, settings.coval_s2s_noisy_persona_id
                    )
                except UnmappedPersona:
                    unmapped.setdefault(run.persona_id, set()).add(f"{slug}/{spec.model}")
                    continue
                counts[condition.value] = counts.get(condition.value, 0) + 1
            flags.extend(
                f"{slug}/{spec.model}: no {condition} runs"
                for condition, count in counts.items()
                if count == 0
            )
            found.append(ModelRuns(slug, spec.model, counts))
    flags.extend(
        f"unmapped persona {persona!r} in {', '.join(sorted(models))}"
        for persona, models in unmapped.items()
    )
    return found, flags


async def _check(
    settings: Settings, window_seconds: int | None
) -> tuple[list[ModelRuns], list[str]]:
    async with coval_client(settings) as client:
        return await check_scenarios(settings, client, window_seconds=window_seconds)


@click.command(name="check-s2s")
@click.option(
    "--settings-file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON of env var names to values, e.g. the fetch job's env from an infra plan.",
)
@click.option(
    "--window-hours",
    type=click.IntRange(min=1),
    help="How far back to look for runs. Defaults to the scheduled fetch's own window.",
)
def check_s2s(settings_file: str | None, window_hours: int | None) -> None:
    """Check the S2S fetch settings against Coval; exits non-zero on any flag."""
    if settings_file:
        with open(settings_file, encoding="utf-8") as f:
            values = json.load(f)
        os.environ.update(
            {
                name: v if isinstance(v, str) else json.dumps(v)
                for name, v in values.items()
                if v is not None
            }
        )
    window_seconds = window_hours * 3600 if window_hours else None
    try:
        found, flags = asyncio.run(_check(Settings(), window_seconds))
    except httpx.HTTPError as exc:
        raise click.ClickException(f"the Coval request failed: {exc}") from exc
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(f"the fetch would reject these settings: {exc}") from exc
    for row in found:
        runs = "  ".join(f"{c}={n}" for c, n in row.runs_by_condition.items())
        click.echo(f"{row.scenario:<16} {row.model:<32} {runs}")
    for flag in flags:
        click.echo(f"FLAG {flag}", err=True)
    if flags:
        raise SystemExit(1)
