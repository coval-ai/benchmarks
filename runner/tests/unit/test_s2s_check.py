# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the read-only S2S settings check."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest
from click.testing import CliRunner

from coval_bench.config import ScenarioCovalIds, Settings
from coval_bench.s2s.check import check_s2s, check_scenarios


def _settings(**kwargs: Any) -> Settings:
    values: dict[str, Any] = {
        "benchmark_artifact_bucket": "test-artifacts",
        "coval_s2s_bank_instruction_metric_id": "JUDGE",
        "coval_s2s_scenarios": {
            "bank": ScenarioCovalIds(
                test_set_id="TSB",
                agents={"gpt-realtime": "A1", "gemini-live": "A2"},
                personas={"clean": "P0", "low": "P1"},
            )
        },
    }
    values.update(kwargs)
    return Settings(**values)


def _client(runs_by_agent: dict[str, list[dict[str, Any]]]) -> httpx.AsyncClient:
    now = datetime.now(tz=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        match = re.search(r'agent_id="([^"]+)"', request.url.params["filter"])
        assert match is not None
        runs = [
            {"run_id": r["run_id"], "persona_id": r["persona_id"], "create_time": now.isoformat()}
            for r in runs_by_agent.get(match.group(1), [])
        ]
        return httpx.Response(200, json={"runs": runs})

    return httpx.AsyncClient(base_url="https://coval.test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_every_model_and_condition_found_has_no_flags() -> None:
    runs = {
        agent: [
            {"run_id": f"{agent}-c", "persona_id": "P0"},
            {"run_id": f"{agent}-l", "persona_id": "P1"},
        ]
        for agent in ("A1", "A2")
    }
    async with _client(runs) as client:
        found, flags = await check_scenarios(_settings(), client, window_seconds=86_400)

    assert flags == []
    assert [(r.model, r.runs_by_condition) for r in found] == [
        ("gpt-realtime", {"clean": 1, "low": 1}),
        ("gemini-live", {"clean": 1, "low": 1}),
    ]


@pytest.mark.asyncio
async def test_model_without_runs_is_flagged_per_condition() -> None:
    runs = {"A1": [{"run_id": "r1", "persona_id": "P0"}, {"run_id": "r2", "persona_id": "P1"}]}
    async with _client(runs) as client:
        _found, flags = await check_scenarios(_settings(), client, window_seconds=86_400)

    assert flags == ["bank/gemini-live: no clean runs", "bank/gemini-live: no low runs"]


@pytest.mark.asyncio
async def test_unmapped_persona_is_flagged() -> None:
    runs = {
        agent: [
            {"run_id": f"{agent}-c", "persona_id": "P0"},
            {"run_id": f"{agent}-l", "persona_id": "P1"},
        ]
        for agent in ("A1", "A2")
    }
    runs["A1"].append({"run_id": "stray-1", "persona_id": "PX"})
    runs["A1"].append({"run_id": "stray-2", "persona_id": "PX"})
    async with _client(runs) as client:
        _found, flags = await check_scenarios(_settings(), client, window_seconds=86_400)

    assert flags == ["unmapped persona 'PX' in bank/gpt-realtime"]


@pytest.mark.asyncio
async def test_unset_judge_is_flagged_as_skipped_by_the_fetch() -> None:
    async with _client({}) as client:
        found, flags = await check_scenarios(
            _settings(coval_s2s_bank_instruction_metric_id=None), client
        )

    assert found == []
    assert flags == [
        "bank/gpt-realtime: the fetch skips it, coval_s2s_bank_instruction_metric_id is unset",
        "bank/gemini-live: the fetch skips it, coval_s2s_bank_instruction_metric_id is unset",
    ]


def test_settings_file_the_fetch_would_reject_exits_non_zero(tmp_path: Path) -> None:
    path = tmp_path / "env.json"
    path.write_text(
        json.dumps(
            {
                "COVAL_S2S_SCENARIOS": {
                    "bank": {
                        "test_set_id": "TSB",
                        "agents": {"not-a-model": "A1"},
                        "personas": {"clean": "P0"},
                    }
                }
            }
        )
    )
    with mock.patch.dict(os.environ, {"COVAL_API_KEY": "test-key"}):
        result = CliRunner().invoke(check_s2s, ["--settings-file", str(path)])

    assert result.exit_code == 1
    assert "the fetch would reject these settings" in result.output
    assert "not-a-model" in result.output
