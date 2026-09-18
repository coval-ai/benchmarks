# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Give Coval its copy of each mock tool call, as a span.

Coval grades only what it holds, and for a phone agent it holds audio and a
transcript: the tool calls the agent made never pass through it. The router
records every call in Postgres (the audit copy). This module posts the same
call to Coval's trace endpoint, addressed with the simulation id the platform
forwarded, so the judge's trace context and the ``tool_calls`` table Coval
exposes to SQL metrics fill in with what the agent actually did.

One span per call, named ``llm_tool_call`` as Coval's convention asks, with
the tool name, its arguments and its result as attributes. Coval flattens each
span into one ``tool_calls`` row, so the attribute names here are a contract
with the SQL metrics that read that table. Batched calls (one platform request
carrying several tool calls) keep identical timestamps and carry their position
in the batch instead: parallel calls are a platform behaviour worth grading,
not a wrinkle to smooth over.

Exporting is best effort. A failure here is logged and never becomes a tool
failure, because a graded conversation must not break on our observability.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import structlog
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from opentelemetry.sdk.resources import Resource

    from coval_bench.mocktools.codecs import ToolCall
    from coval_bench.mocktools.dispatch import Outcome

logger = structlog.get_logger("coval_bench.mocktools.traces")

SPAN_NAME = "llm_tool_call"
TRACES_ENDPOINT = "https://api.coval.dev/v1/traces"
SIMULATION_HEADER = "X-Simulation-Id"
# Coval documents 30 s as the required exporter timeout.
EXPORT_TIMEOUT_S = 30


def coval_exporter(api_key: str, simulation_id: str) -> OTLPSpanExporter:
    """An exporter addressed to one simulation.

    Coval reads the simulation id from a request header, not from the span, and
    an exporter's headers are fixed when it is built. So each simulation gets its
    own exporter; at a handful of tool calls per conversation that is nothing.
    """
    return OTLPSpanExporter(
        endpoint=TRACES_ENDPOINT,
        headers={"X-API-Key": api_key, SIMULATION_HEADER: simulation_id},
        timeout=EXPORT_TIMEOUT_S,
    )


def span_attributes(
    call: ToolCall, outcome: Outcome, platform: str, index: int, batch_size: int
) -> dict[str, AttributeValue]:
    """The one span's attributes: Coval's tool-call convention plus our provenance."""
    attributes: dict[str, AttributeValue] = {
        "function.name": call.tool,
        "function.arguments": json.dumps(call.args, sort_keys=True),
        "tool.result": json.dumps(outcome.response, sort_keys=True),
        "coval_bench.platform": platform,
        "coval_bench.resolution": outcome.resolution.mode if outcome.resolution else "rejected",
        "coval_bench.batch_index": index,
        "coval_bench.batch_size": batch_size,
    }
    if call.call_id:
        attributes["tool_call_id"] = call.call_id
    if outcome.resolution and outcome.resolution.matched_seed:
        attributes["coval_bench.seed"] = outcome.resolution.matched_seed
    error = outcome.response.get("error")
    if error is not None:
        attributes["tool.error"] = str(error)
    elif outcome.http_status >= 400:
        attributes["tool.error"] = f"http_{outcome.http_status}"
    return attributes


def export_tool_spans(
    exporter: SpanExporter,
    resource: Resource,
    platform: str,
    calls: Sequence[ToolCall],
    outcomes: Sequence[Outcome],
    started_ns: int,
    ended_ns: int,
) -> int:
    """Post one span per call and return how many were handed to the exporter.

    Never raises: the spans are a copy for grading, and losing the copy must not
    turn into anything the agent under test can observe.
    """
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    try:
        for index, (call, outcome) in enumerate(zip(calls, outcomes, strict=True)):
            span = tracer.start_span(SPAN_NAME, start_time=started_ns)
            span.set_attributes(span_attributes(call, outcome, platform, index, len(calls)))
            span.end(end_time=ended_ns)
    except Exception:
        logger.warning("mock_tool_spans_failed", platform=platform, exc_info=True)
        return 0
    finally:
        provider.shutdown()
    return len(calls)
