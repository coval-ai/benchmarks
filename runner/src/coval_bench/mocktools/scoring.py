# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a checked tool's first call and give Coval the score, the way Corral does.

Corral checks the first call of each checked tool when its own call ends. A
platform's call never ends here, so the first call this process sees for a
simulation is scored instead, after the answer has gone back. The span carries
the same name and attributes Corral's does, so one Coval metric reads both.

"First" is per process: two instances that each see a call for one simulation
both score it. Agents are told to call once, so this only matters when they don't.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

import httpx
import structlog
from opentelemetry.sdk.trace.export import SpanExportResult

from coval_bench.mocktools.tool_checks import CHARACTER_ERROR_RATE, character_error_rates
from coval_bench.mocktools.traces import build_check_span, post_spans, queue

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from pydantic import SecretStr

    from coval_bench.mocktools.codecs import ToolCall
    from coval_bench.mocktools.dispatch import Check

logger = structlog.get_logger("coval_bench.mocktools.scoring")

TEST_CASE_ATTRIBUTE = "test_case"
LOOKUP_TIMEOUT_S = 10.0
MAX_CONCURRENT_LOOKUPS = 8
MAX_REMEMBERED = 10_000

_scored: OrderedDict[tuple[str, str], None] = OrderedDict()
_lookups = asyncio.Semaphore(MAX_CONCURRENT_LOOKUPS)


def claim_first_calls(
    simulation_id: str, calls: Sequence[ToolCall], checked: Collection[str]
) -> list[ToolCall]:
    """Each checked tool's first call this process has seen for the simulation."""
    claimed: list[ToolCall] = []
    for call in calls:
        key = (simulation_id, call.tool)
        if call.tool not in checked or key in _scored:
            continue
        _scored[key] = None
        if len(_scored) > MAX_REMEMBERED:
            _scored.popitem(last=False)
        claimed.append(call)
    return claimed


def release(simulation_id: str, calls: Sequence[ToolCall]) -> None:
    """Give back claims whose score never reached Coval, so a later call can be scored."""
    for call in calls:
        _scored.pop((simulation_id, call.tool), None)


def right_values(check: Check, test_case: str | None) -> dict[str, str]:
    """Each checked argument's right value for this test case; per-case ones need a case."""
    values: dict[str, str] = {}
    for argument, value in check.items():
        if isinstance(value, str):
            values[argument] = value
        elif test_case is not None and test_case in value:
            values[argument] = value[test_case]
    return values


async def test_case_key(api_key: str, api_base: str, simulation_id: str) -> str | None:
    """The calling test case's ``test_case`` attribute in Coval, or None when it has none."""
    async with (
        _lookups,
        httpx.AsyncClient(
            base_url=api_base, headers={"X-API-Key": api_key}, timeout=LOOKUP_TIMEOUT_S
        ) as client,
    ):
        response = await client.get(
            f"/simulations/{simulation_id}/attributes/{TEST_CASE_ATTRIBUTE}"
        )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, str):
        raise ValueError(f"the test case's {TEST_CASE_ATTRIBUTE} is {value!r}, not a name")
    return value


async def _score(
    api_key: str,
    api_base: str,
    provider: TracerProvider,
    simulation_id: str,
    platform: str,
    checks: Mapping[str, Check],
    calls: Sequence[ToolCall],
    started_ns: int,
    ended_ns: int,
) -> None:
    try:
        per_case = any(isinstance(v, dict) for call in calls for v in checks[call.tool].values())
        test_case = await test_case_key(api_key, api_base, simulation_id) if per_case else None
        expected: dict[str, dict[str, str]] = {}
        for call in calls:
            values = right_values(checks[call.tool], test_case)
            if len(values) < len(checks[call.tool]):
                logger.warning(
                    "mock_tool_check_unscored",
                    simulation_id=simulation_id,
                    tool=call.tool,
                    test_case=test_case,
                )
                continue
            expected[call.tool] = values
        if not expected:
            return
        rates = character_error_rates(expected, calls)
        span = build_check_span(
            provider, CHARACTER_ERROR_RATE, platform, rates, test_case, started_ns, ended_ns
        )
        result = await post_spans(api_key, api_base, simulation_id, [span])
    except asyncio.CancelledError:
        release(simulation_id, calls)
        raise
    except Exception:
        release(simulation_id, calls)
        logger.warning(
            "mock_tool_check_failed", simulation_id=simulation_id, platform=platform, exc_info=True
        )
        return
    if result is SpanExportResult.SUCCESS:
        logger.info(
            "mock_tool_check_exported", simulation_id=simulation_id, platform=platform, **rates
        )
    else:
        release(simulation_id, calls)
        logger.warning("mock_tool_check_failed", simulation_id=simulation_id, platform=platform)


def schedule_scoring(
    *,
    api_key: SecretStr | None,
    api_base: str,
    provider: TracerProvider,
    simulation_id: str,
    platform: str,
    checks: Mapping[str, Check],
    calls: Sequence[ToolCall],
    started_ns: int,
    ended_ns: int,
) -> bool:
    """Queue the score for each checked tool's first call; returns whether anything was queued."""
    if api_key is None:
        return False
    claimed = claim_first_calls(simulation_id, calls, checks)
    if not claimed:
        return False
    queued = queue(
        _score(
            api_key.get_secret_value(),
            api_base,
            provider,
            simulation_id,
            platform,
            checks,
            claimed,
            started_ns,
            ended_ns,
        ),
        simulation_id=simulation_id,
        platform=platform,
        count=len(claimed),
    )
    if not queued:
        release(simulation_id, claimed)
    return queued
