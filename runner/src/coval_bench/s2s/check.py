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

from coval_bench.config import Settings, get_settings
from coval_bench.s2s.fetch_v2v import (
    UnmappedPersona,
    _all_persona_conditions,
    _client,
    _condition_for_persona,
    _persona_conditions,
    _scenario_specs,
    recent_completed_runs,
)


@dataclass(frozen=True)
class ModelRuns:
    scenario: str
    model: str
    runs_by_condition: dict[str, int]


async def check_scenarios(
    settings: Settings, client: httpx.AsyncClient, *, window_seconds: int
) -> tuple[list[ModelRuns], list[str]]:
    """Count each scenario model's recent runs per condition and flag what the fetch would miss."""
    persona_conditions = _persona_conditions(_all_persona_conditions(settings))
    specs = _scenario_specs(settings)
    found: list[ModelRuns] = []
    flags: list[str] = []
    for slug, ids in settings.coval_s2s_scenarios.items():
        scenario_specs = [s for s in specs if s.test_set_id == ids.test_set_id and s.agent_id]
        if not scenario_specs:
            flags.append(f"{slug}: the fetch skips this scenario (no agents or no judge)")
            continue
        for spec in scenario_specs:
            if spec.agent_id is None:
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
                    condition = _condition_for_persona(run.persona_id, persona_conditions, None)
                except UnmappedPersona:
                    flags.append(
                        f"{slug}/{spec.model}: run {run.run_id} has unmapped persona "
                        f"{run.persona_id!r}"
                    )
                    continue
                counts[condition.value] = counts.get(condition.value, 0) + 1
            flags.extend(
                f"{slug}/{spec.model}: no {condition} runs"
                for condition, count in counts.items()
                if count == 0
            )
            found.append(ModelRuns(slug, spec.model, counts))
    return found, flags


async def _check(settings: Settings, window_seconds: int) -> tuple[list[ModelRuns], list[str]]:
    async with _client(settings) as client:
        return await check_scenarios(settings, client, window_seconds=window_seconds)


@click.command(name="check-s2s")
@click.option(
    "--settings-file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON of env var names to values, e.g. the fetch job's env from an infra plan.",
)
@click.option("--window-hours", default=36, show_default=True, type=click.IntRange(min=1))
def check_s2s(settings_file: str | None, window_hours: int) -> None:
    """Check the S2S fetch settings against Coval; exits non-zero on any flag."""
    if settings_file:
        with open(settings_file, encoding="utf-8") as f:
            values = json.load(f)
        os.environ.update(
            {name: v if isinstance(v, str) else json.dumps(v) for name, v in values.items()}
        )
    try:
        found, flags = asyncio.run(_check(get_settings(), window_hours * 3600))
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(f"the fetch would reject these settings: {exc}") from exc
    for row in found:
        runs = "  ".join(f"{c}={n}" for c, n in row.runs_by_condition.items())
        click.echo(f"{row.scenario:<16} {row.model:<32} {runs}")
    for flag in flags:
        click.echo(f"FLAG {flag}", err=True)
    if flags:
        raise SystemExit(1)
