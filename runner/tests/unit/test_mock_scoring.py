# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Scoring a checked tool's first call and posting the score to Coval as Corral's span."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from coval_bench.mocktools import scoring
from coval_bench.mocktools.codecs import ToolCall
from coval_bench.mocktools.dispatch import Check
from coval_bench.mocktools.traces import new_tracer_provider

SIM = "sim_1"
CHECKS: dict[str, Check] = {"lookup": {"value": {"account-1": "KD7BO0RWNLMX0O4BYD"}}}


@pytest.fixture(autouse=True)
def _forget() -> Iterator[None]:
    scoring._scored.clear()
    yield
    scoring._scored.clear()


def test_only_each_checked_tools_first_call_per_simulation_is_claimed() -> None:
    first, retry = ToolCall("lookup", {"value": "a"}), ToolCall("lookup", {"value": "b"})
    other = ToolCall("transfer", {})
    assert scoring.claim_first_calls(SIM, [first, other], CHECKS) == [first]
    assert scoring.claim_first_calls(SIM, [retry], CHECKS) == []
    assert scoring.claim_first_calls("sim_2", [retry], CHECKS) == [retry]


def test_remembered_simulations_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scoring, "MAX_REMEMBERED", 2)
    for sim in ("a", "b", "c"):
        scoring.claim_first_calls(sim, [ToolCall("lookup", {})], CHECKS)
    assert list(scoring._scored) == [("b", "lookup"), ("c", "lookup")]


def test_right_values_take_the_shared_value_or_the_test_cases_own() -> None:
    check: Check = {"value": {"account-1": "A1"}, "kind": "card"}
    assert scoring.right_values(check, "account-1") == {"value": "A1", "kind": "card"}
    assert scoring.right_values(check, None) == {"kind": "card"}
    assert scoring.right_values(check, "unknown") == {"kind": "card"}


@pytest.fixture
def posted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[ReadableSpan]]]:
    seen: list[tuple[str, list[ReadableSpan]]] = []

    async def fake_post(
        api_key: str, api_base: str, simulation_id: str, spans: list[ReadableSpan]
    ) -> SpanExportResult:
        seen.append((simulation_id, list(spans)))
        return SpanExportResult.SUCCESS

    monkeypatch.setattr(scoring, "post_spans", fake_post)
    return seen


def _lookup_answers(monkeypatch: pytest.MonkeyPatch, test_case: str | None) -> list[str]:
    asked: list[str] = []

    async def fake_lookup(api_key: str, api_base: str, simulation_id: str) -> str | None:
        asked.append(simulation_id)
        return test_case

    monkeypatch.setattr(scoring, "test_case_key", fake_lookup)
    return asked


async def _score(calls: list[ToolCall], checks: dict[str, Check] = CHECKS) -> None:
    provider = new_tracer_provider(Resource.create({"service.name": "test"}))
    await scoring._score("key", "https://coval.test/v1", provider, SIM, "vapi", checks, calls, 1, 2)


async def test_the_score_goes_to_coval_as_corrals_span(
    monkeypatch: pytest.MonkeyPatch, posted: list[tuple[str, list[ReadableSpan]]]
) -> None:
    asked = _lookup_answers(monkeypatch, "account-1")
    await _score([ToolCall("lookup", {"value": "KD7B00RWNLMX0O4BYD"})])
    assert asked == [SIM]
    [(simulation_id, [span])] = posted
    assert simulation_id == SIM
    assert span.name == "character_error_rate"
    attributes: dict[str, Any] = dict(span.attributes or {})
    assert attributes["lookup.value"] == pytest.approx(1 / 18)
    assert attributes["coval_bench.platform"] == "vapi"
    assert attributes["coval_bench.test_case"] == "account-1"


async def test_a_shared_right_value_never_asks_coval_for_the_test_case(
    monkeypatch: pytest.MonkeyPatch, posted: list[tuple[str, list[ReadableSpan]]]
) -> None:
    asked = _lookup_answers(monkeypatch, "account-1")
    await _score([ToolCall("lookup", {"value": "x"})], {"lookup": {"value": "x"}})
    assert asked == []
    assert dict(posted[0][1][0].attributes or {})["lookup.value"] == 0.0


async def test_an_unknown_test_case_posts_nothing(
    monkeypatch: pytest.MonkeyPatch, posted: list[tuple[str, list[ReadableSpan]]]
) -> None:
    _lookup_answers(monkeypatch, "account-9")
    await _score([ToolCall("lookup", {"value": "x"})])
    assert posted == []


async def test_a_failed_lookup_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, posted: list[tuple[str, list[ReadableSpan]]]
) -> None:
    async def broken(api_key: str, api_base: str, simulation_id: str) -> str | None:
        raise RuntimeError("coval down")

    monkeypatch.setattr(scoring, "test_case_key", broken)
    await _score([ToolCall("lookup", {"value": "x"})])
    assert posted == []


def test_nothing_is_scheduled_without_a_coval_key() -> None:
    provider = new_tracer_provider(Resource.create({"service.name": "test"}))
    queued = scoring.schedule_scoring(
        api_key=None,
        api_base="https://coval.test/v1",
        provider=provider,
        simulation_id=SIM,
        platform="vapi",
        checks=CHECKS,
        calls=[ToolCall("lookup", {"value": "x"})],
        started_ns=1,
        ended_ns=2,
    )
    assert queued is False
    assert not scoring._scored
