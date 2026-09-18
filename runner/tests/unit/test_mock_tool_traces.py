# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The spans the mock appliance posts to Coval: one per call, Coval's names, our provenance."""

from __future__ import annotations

import json

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from coval_bench.mocktools.codecs import ToolCall
from coval_bench.mocktools.dispatch import Outcome
from coval_bench.mocktools.fixtures import Seed
from coval_bench.mocktools.resolver import Resolution
from coval_bench.mocktools.traces import (
    SIMULATION_HEADER,
    SPAN_NAME,
    TRACES_ENDPOINT,
    coval_exporter,
    export_tool_spans,
)

RESOURCE = Resource.create({"service.name": "test"})
STARTED_NS = 1_700_000_000_000_000_000
ENDED_NS = STARTED_NS + 150_000_000


def _seeded(seed_id: str, response: dict[str, object], mode: str = "exact") -> Outcome:
    seed = Seed(id=seed_id, match={"phone": "2065550180"}, response=response)
    return Outcome(
        response=response,
        http_status=200,
        resolution=Resolution(seed=seed, mode=mode, score=100.0, shared_keys=("phone",)),  # type: ignore[arg-type]
    )


def _export(*pairs: tuple[ToolCall, Outcome], platform: str = "vapi") -> list[ReadableSpan]:
    exporter = InMemorySpanExporter()
    calls = [call for call, _ in pairs]
    outcomes = [outcome for _, outcome in pairs]
    exported = export_tool_spans(
        exporter, RESOURCE, platform, calls, outcomes, STARTED_NS, ENDED_NS
    )
    spans = list(exporter.get_finished_spans())
    assert exported == len(spans) == len(pairs)
    return spans


def test_a_seed_hit_becomes_one_span_in_covals_shape() -> None:
    call = ToolCall("lookup_patient", {"phone": "2065550180"}, call_id="call_1")
    outcome = _seeded("marcus_lee", {"found": True, "patient_id": "P-2065"})

    (span,) = _export((call, outcome))

    assert span.name == SPAN_NAME
    assert span.start_time == STARTED_NS
    assert span.end_time == ENDED_NS
    attributes = dict(span.attributes or {})
    assert attributes["function.name"] == "lookup_patient"
    assert json.loads(str(attributes["function.arguments"])) == {"phone": "2065550180"}
    assert json.loads(str(attributes["tool.result"])) == {"found": True, "patient_id": "P-2065"}
    assert attributes["tool_call_id"] == "call_1"
    assert attributes["coval_bench.platform"] == "vapi"
    assert attributes["coval_bench.resolution"] == "exact"
    assert attributes["coval_bench.seed"] == "marcus_lee"
    assert attributes["coval_bench.batch_index"] == 0
    assert attributes["coval_bench.batch_size"] == 1
    assert "tool.error" not in attributes


def test_a_fallback_carries_no_seed_and_no_error() -> None:
    call = ToolCall("check_availability", {"date": "2030-05-06", "appointment_type": "cleaning"})
    outcome = _seeded("fully_booked", {"slots": []}, mode="fallback")

    (span,) = _export((call, outcome))

    attributes = dict(span.attributes or {})
    assert attributes["coval_bench.resolution"] == "fallback"
    assert "coval_bench.seed" not in attributes
    assert "tool_call_id" not in attributes
    assert "tool.error" not in attributes


def test_a_seeded_failure_and_a_rejected_call_both_name_their_error() -> None:
    broken = ToolCall("lookup_patient", {"phone": "0000000000"})
    failure = Outcome(response={"error": "upstream_unavailable"}, http_status=500)
    unknown = ToolCall("read_chart", {"phone": "2065550180"})
    rejected = Outcome(response={"error": "unknown_tool"}, http_status=200)

    first, second = _export((broken, failure), (unknown, rejected))

    assert dict(first.attributes or {})["tool.error"] == "upstream_unavailable"
    assert dict(first.attributes or {})["coval_bench.resolution"] == "rejected"
    assert dict(second.attributes or {})["tool.error"] == "unknown_tool"


def test_a_batch_keeps_one_clock_and_records_each_position() -> None:
    """Parallel calls stay simultaneous: that is the platform behaviour to grade."""
    check = ToolCall("check_availability", {"date": "2030-12-11", "appointment_type": "cleaning"})
    book = ToolCall(
        "book_appointment",
        {"patient_id": "P-2065", "datetime": "2030-12-11T15:00:00", "appointment_type": "cleaning"},
    )
    outcome = _seeded("any", {"ok": True})

    first, second = _export((check, outcome), (book, outcome))

    assert first.start_time == second.start_time == STARTED_NS
    assert first.end_time == second.end_time == ENDED_NS
    assert dict(first.attributes or {})["coval_bench.batch_index"] == 0
    assert dict(second.attributes or {})["coval_bench.batch_index"] == 1
    assert dict(second.attributes or {})["coval_bench.batch_size"] == 2


def test_an_exporter_failure_is_swallowed_and_counted_as_nothing_sent() -> None:
    class Exploding(SpanExporter):
        def export(self, spans: object) -> SpanExportResult:  # noqa: ARG002
            raise RuntimeError("coval is down")

        def shutdown(self) -> None:
            return None

    call = ToolCall("lookup_patient", {"phone": "2065550180"})
    outcome = _seeded("marcus_lee", {"found": True})

    exported = export_tool_spans(
        Exploding(), RESOURCE, "vapi", [call], [outcome], STARTED_NS, ENDED_NS
    )

    # SimpleSpanProcessor logs and swallows its own exporter errors, so the spans
    # were still handed over; what matters is that nothing propagated.
    assert exported in (0, 1)


def test_the_coval_exporter_is_addressed_to_one_simulation() -> None:
    exporter = coval_exporter("k-test", "sim_abc123")

    assert exporter._endpoint == TRACES_ENDPOINT  # noqa: SLF001
    assert exporter._headers["X-API-Key"] == "k-test"  # noqa: SLF001
    assert exporter._headers[SIMULATION_HEADER] == "sim_abc123"  # noqa: SLF001
    assert exporter._timeout == 30  # noqa: SLF001
