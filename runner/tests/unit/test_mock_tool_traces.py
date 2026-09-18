# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The spans the mock appliance posts to Coval: one trace per request, Coval's names, ours."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

import coval_bench.mocktools.traces as traces
from coval_bench.mocktools.codecs import ToolCall
from coval_bench.mocktools.dispatch import Outcome
from coval_bench.mocktools.fixtures import Seed
from coval_bench.mocktools.resolver import Resolution
from coval_bench.mocktools.traces import (
    REQUEST_SPAN_NAME,
    SIMULATION_HEADER,
    SPAN_NAME,
    build_spans,
    coval_exporter,
    export_spans,
    schedule_export,
    span_attributes,
)

RESOURCE = Resource.create({"service.name": "test"})
STARTED_NS = 1_700_000_000_000_000_000
ENDED_NS = STARTED_NS + 150_000_000
LOOKUP = ToolCall("lookup_patient", {"phone": "2065550180"}, call_id="call_1")


def _seeded(seed_id: str, response: dict[str, Any], mode: str = "exact") -> Outcome:
    seed = Seed(id=seed_id, match={"phone": "2065550180"}, response=response)
    return Outcome(
        response=response,
        http_status=200,
        resolution=Resolution(seed=seed, mode=mode, score=100.0, shared_keys=("phone",)),  # type: ignore[arg-type]
    )


def _build(*pairs: tuple[ToolCall, Outcome], platform: str = "vapi") -> list[ReadableSpan]:
    calls = [call for call, _ in pairs]
    outcomes = [outcome for _, outcome in pairs]
    return build_spans(RESOURCE, platform, calls, outcomes, STARTED_NS, ENDED_NS)


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


# --- one trace per request ---------------------------------------------------


def test_a_request_is_one_trace_with_a_child_per_call() -> None:
    check = ToolCall("check_availability", {"date": "2030-12-11", "appointment_type": "cleaning"})
    outcome = _seeded("any", {"ok": True})

    first, second, request = _build((LOOKUP, outcome), (check, outcome))

    assert request.name == REQUEST_SPAN_NAME
    assert request.parent is None
    assert {first.name, second.name} == {SPAN_NAME}
    assert first.context.trace_id == second.context.trace_id == request.context.trace_id
    assert first.parent is not None and first.parent.span_id == request.context.span_id
    assert second.parent is not None and second.parent.span_id == request.context.span_id
    assert _attributes(request)["coval_bench.batch_size"] == 2
    assert request.status.status_code is StatusCode.UNSET


def test_a_batch_keeps_one_clock_and_records_each_position() -> None:
    """Parallel calls stay simultaneous: that is the platform behaviour to grade."""
    book = ToolCall(
        "book_appointment",
        {"patient_id": "P-2065", "datetime": "2030-12-11T15:00:00", "appointment_type": "cleaning"},
    )
    outcome = _seeded("any", {"ok": True})

    first, second, _request = _build((LOOKUP, outcome), (book, outcome))

    assert first.start_time == second.start_time == STARTED_NS
    assert first.end_time == second.end_time == ENDED_NS
    assert _attributes(first)["coval_bench.batch_index"] == 0
    assert _attributes(second)["coval_bench.batch_index"] == 1
    assert _attributes(second)["coval_bench.batch_size"] == 2


# --- attributes ---------------------------------------------------------------


def test_a_seed_hit_carries_covals_names_and_our_provenance() -> None:
    outcome = _seeded("marcus_lee", {"found": True, "patient_id": "P-2065"})

    (span, _request) = _build((LOOKUP, outcome))

    attributes = _attributes(span)
    assert attributes["function.name"] == "lookup_patient"
    assert json.loads(str(attributes["function.arguments"])) == {"phone": "2065550180"}
    assert json.loads(str(attributes["tool.result"])) == {"found": True, "patient_id": "P-2065"}
    assert attributes["tool_call_id"] == "call_1"
    assert attributes["coval_bench.platform"] == "vapi"
    assert attributes["coval_bench.resolution"] == "exact"
    assert attributes["coval_bench.seed"] == "marcus_lee"
    assert "tool.error" not in attributes
    assert span.status.status_code is StatusCode.UNSET


def test_a_fallback_carries_no_seed_and_no_error() -> None:
    call = ToolCall("check_availability", {"date": "2030-05-06", "appointment_type": "cleaning"})
    outcome = _seeded("fully_booked", {"slots": []}, mode="fallback")

    (span, _request) = _build((call, outcome))

    attributes = _attributes(span)
    assert attributes["coval_bench.resolution"] == "fallback"
    assert "coval_bench.seed" not in attributes
    assert "tool_call_id" not in attributes
    assert "tool.error" not in attributes


def test_failed_calls_carry_the_error_and_an_error_status() -> None:
    failure = Outcome(response={"error": "upstream_unavailable"}, http_status=500)
    rejected = Outcome(response={"error": "unknown_tool"}, http_status=200)

    first, second, request = _build((LOOKUP, failure), (ToolCall("read_chart", {}), rejected))

    assert _attributes(first)["tool.error"] == "upstream_unavailable"
    assert _attributes(first)["coval_bench.resolution"] == "rejected"
    assert first.status.status_code is StatusCode.ERROR
    assert _attributes(second)["tool.error"] == "unknown_tool"
    assert second.status.status_code is StatusCode.ERROR
    assert request.status.status_code is StatusCode.ERROR
    assert request.status.description == "2 of 2 tool calls failed"


def test_a_status_failure_without_an_error_body_is_still_an_error() -> None:
    outcome = Outcome(response={"detail": "boom"}, http_status=503)

    (span, _request) = _build((LOOKUP, outcome))

    assert _attributes(span)["tool.error"] == "http_503"
    assert span.status.status_code is StatusCode.ERROR


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (False, None),
        ("", None),
        ({"code": "E_NOT_FOUND"}, '{"code": "E_NOT_FOUND"}'),
        ("plain", "plain"),
    ],
)
def test_tool_error_is_set_only_for_a_real_error_and_kept_as_json(
    error: object, expected: str | None
) -> None:
    outcome = Outcome(response={"error": error, "found": False}, http_status=200)

    attributes = span_attributes(LOOKUP, outcome, "vapi", 0, 1)

    assert attributes.get("tool.error") == expected


# --- export ---------------------------------------------------------------------


def test_a_batch_is_one_export_holding_every_span() -> None:
    exporter = InMemorySpanExporter()
    spans = _build((LOOKUP, _seeded("marcus_lee", {"found": True})))

    result = export_spans(exporter, spans)

    assert result is SpanExportResult.SUCCESS
    assert len(exporter.get_finished_spans()) == 2


class _Exploding(SpanExporter):
    def __init__(self, *, on_shutdown: bool = False) -> None:
        self.calls = 0
        self.on_shutdown = on_shutdown

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.calls += 1
        raise RuntimeError("coval is down")

    def shutdown(self) -> None:
        if self.on_shutdown:
            raise RuntimeError("session already closed")


def test_an_exporter_that_raises_is_reported_as_a_failure_not_an_exception() -> None:
    exporter = _Exploding()
    spans = _build((LOOKUP, _seeded("marcus_lee", {"found": True})))

    result = export_spans(exporter, spans)

    assert result is SpanExportResult.FAILURE
    assert exporter.calls == 1


def test_a_shutdown_that_raises_does_not_escape_either() -> None:
    exporter = _Exploding(on_shutdown=True)
    spans = _build((LOOKUP, _seeded("marcus_lee", {"found": True})))

    assert export_spans(exporter, spans) is SpanExportResult.FAILURE


def test_the_coval_exporter_is_addressed_to_one_simulation_beside_the_api() -> None:
    exporter = coval_exporter("k-test", "sim_abc123", "https://staging.coval.invalid/v1/")

    assert exporter._endpoint == "https://staging.coval.invalid/v1/traces"  # noqa: SLF001
    assert exporter._headers["X-API-Key"] == "k-test"  # noqa: SLF001
    assert exporter._headers[SIMULATION_HEADER] == "sim_abc123"  # noqa: SLF001
    assert exporter._timeout == 30  # noqa: SLF001


# --- scheduling -------------------------------------------------------------------


async def test_without_a_key_nothing_is_queued_and_the_process_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(traces, "_disabled_warned", False)
    warnings: list[str] = []
    monkeypatch.setattr(traces.logger, "warning", lambda event, **_kw: warnings.append(str(event)))
    common = _common()

    assert schedule_export(api_key=None, **common) is False
    assert schedule_export(api_key=None, **common) is False
    assert warnings == ["mock_tool_traces_disabled"]
    assert not traces._tasks  # noqa: SLF001


async def test_with_a_key_the_batch_is_posted_off_the_request_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: list[tuple[str, str, int]] = []

    def fake_export(exporter: Any, spans: Sequence[ReadableSpan]) -> SpanExportResult:  # noqa: ANN401
        posted.append(
            (exporter._headers[SIMULATION_HEADER], exporter._endpoint, len(spans))  # noqa: SLF001
        )
        return SpanExportResult.SUCCESS

    monkeypatch.setattr(traces, "export_spans", fake_export)

    queued = schedule_export(api_key=SecretStr("k-test"), **_common())
    await asyncio.gather(*traces._tasks)  # noqa: SLF001

    assert queued is True
    assert posted == [("sim_abc123", "https://coval.invalid/v1/traces", 2)]


async def test_the_pending_queue_is_bounded_and_overflow_is_dropped_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()

    async def held_export(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        await gate.wait()

    monkeypatch.setattr(traces, "_export", held_export)
    monkeypatch.setattr(traces, "MAX_PENDING_EXPORTS", 2)
    warnings: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        traces.logger, "warning", lambda event, **kw: warnings.append((str(event), kw))
    )
    key = SecretStr("k-test")

    assert schedule_export(api_key=key, **_common()) is True
    assert schedule_export(api_key=key, **_common()) is True
    assert schedule_export(api_key=key, **_common()) is False
    assert len(traces._tasks) == 2  # noqa: SLF001
    assert warnings == [
        (
            "mock_tool_spans_dropped",
            {"simulation_id": "sim_abc123", "platform": "vapi", "count": 1, "pending": 2},
        )
    ]

    gate.set()
    await asyncio.gather(*traces._tasks)  # noqa: SLF001


async def test_drain_waits_for_queued_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    finished = asyncio.Event()

    async def quick_export(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        await asyncio.sleep(0)
        finished.set()

    monkeypatch.setattr(traces, "_export", quick_export)
    schedule_export(api_key=SecretStr("k-test"), **_common())

    assert await traces.drain(timeout_s=1) == 0
    assert finished.is_set()
    assert not traces._tasks  # noqa: SLF001


async def test_drain_abandons_a_stuck_export_after_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stuck_export(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        await asyncio.Event().wait()

    monkeypatch.setattr(traces, "_export", stuck_export)
    warnings: list[str] = []
    monkeypatch.setattr(traces.logger, "warning", lambda event, **_kw: warnings.append(str(event)))
    schedule_export(api_key=SecretStr("k-test"), **_common())

    assert await traces.drain(timeout_s=0.01) == 1
    assert warnings == ["mock_tool_spans_abandoned"]
    assert not traces._tasks  # noqa: SLF001


def _common() -> dict[str, Any]:
    return {
        "api_base": "https://coval.invalid/v1",
        "resource": lambda: RESOURCE,
        "simulation_id": "sim_abc123",
        "platform": "vapi",
        "calls": [LOOKUP],
        "outcomes": [_seeded("marcus_lee", {"found": True})],
        "started_ns": STARTED_NS,
        "ended_ns": ENDED_NS,
    }
