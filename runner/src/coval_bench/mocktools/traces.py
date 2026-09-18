# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Give Coval its copy of each mock tool call, as a span.

Coval grades only what it holds, and for a phone agent it holds audio and a
transcript: the tool calls the agent made never pass through it. The router
records every call in Postgres (the audit copy). This module posts the same
call to Coval's trace endpoint, addressed with the simulation id the platform
forwarded, so the judge's trace context and the ``tool_calls`` table Coval
exposes to SQL metrics fill in with what the agent actually did.

One request to the appliance becomes one trace: a ``mock_tool_request`` parent
and one ``llm_tool_call`` child per call, the name Coval's convention asks for,
carrying the tool name, its arguments and its result. Coval flattens each child
into one ``tool_calls`` row, so the attribute names here are a contract with
the SQL metrics that read that table. Calls in one batch keep identical
timestamps and carry their position instead: parallel calls are a platform
behaviour worth grading, not a wrinkle to smooth over. A call that failed
carries ``tool.error`` and an ERROR status, so status-based metrics see it.

The copy is posted off the request path. Nothing here runs before the tool
answer is sent, the work happens on an event-loop task rather than the thread
pool the router's dependencies share, at most :data:`MAX_CONCURRENT_EXPORTS`
posts are in flight, and one batch is one POST bounded by the exporter's own
timeout. A failure is logged with the simulation id and never becomes a tool
failure, because a graded conversation must not break on our observability.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from pydantic import SecretStr

    from coval_bench.mocktools.codecs import ToolCall
    from coval_bench.mocktools.dispatch import Outcome

logger = structlog.get_logger("coval_bench.mocktools.traces")

SERVICE_NAME = "benchmarks-mock-tools"
REQUEST_SPAN_NAME = "mock_tool_request"
SPAN_NAME = "llm_tool_call"
SIMULATION_HEADER = "X-Simulation-Id"
# Coval documents 30 s as the required exporter timeout. The exporter's retry
# loop for 5xx and connection errors runs inside this budget, so it is also the
# most one batch can hold a thread.
EXPORT_TIMEOUT_S = 30
MAX_CONCURRENT_EXPORTS = 8

# One provider for the process, with no span processor attached: spans are
# collected by hand and posted as a batch, rather than exported one by one as
# they end. Building a provider per request would also register a fork hook
# the interpreter never releases.
_provider: TracerProvider | None = None
_exports = asyncio.Semaphore(MAX_CONCURRENT_EXPORTS)
_tasks: set[asyncio.Task[None]] = set()
_disabled_warned = False


def _tracer(resource: Resource) -> trace.Tracer:
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=resource)
    return _provider.get_tracer(__name__)


def _readable(span: trace.Span) -> ReadableSpan:
    if not isinstance(span, ReadableSpan):  # pragma: no cover — the SDK provider always is
        raise TypeError(f"expected an SDK span, got {type(span).__name__}")
    return span


def traces_endpoint(api_base: str) -> str:
    """Coval's OTLP ingest, beside the REST API the rest of the runner talks to."""
    return api_base.rstrip("/") + "/traces"


def coval_exporter(api_key: str, simulation_id: str, api_base: str) -> OTLPSpanExporter:
    """An exporter addressed to one simulation.

    Coval reads the simulation id from a request header, not from the span, and
    an exporter's headers are fixed when it is built. So each batch gets its own
    exporter; it is a small object and its session closes with it.
    """
    return OTLPSpanExporter(
        endpoint=traces_endpoint(api_base),
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
        "coval_bench.resolution": outcome.mode,
        "coval_bench.batch_index": index,
        "coval_bench.batch_size": batch_size,
    }
    if call.call_id:
        attributes["tool_call_id"] = call.call_id
    if outcome.matched_seed:
        attributes["coval_bench.seed"] = outcome.matched_seed
    error = outcome.response.get("error")
    if error:
        attributes["tool.error"] = (
            error if isinstance(error, str) else json.dumps(error, sort_keys=True)
        )
    elif outcome.http_status >= 400:
        attributes["tool.error"] = f"http_{outcome.http_status}"
    return attributes


def build_spans(
    resource: Resource,
    platform: str,
    calls: Sequence[ToolCall],
    outcomes: Sequence[Outcome],
    started_ns: int,
    ended_ns: int,
) -> list[ReadableSpan]:
    """One trace for the request: a parent span with one child per tool call."""
    tracer = _tracer(resource)
    request = tracer.start_span(
        REQUEST_SPAN_NAME,
        start_time=started_ns,
        attributes={"coval_bench.platform": platform, "coval_bench.batch_size": len(calls)},
    )
    context = trace.set_span_in_context(request)
    spans: list[ReadableSpan] = []
    failed = 0
    for index, (call, outcome) in enumerate(zip(calls, outcomes, strict=True)):
        attributes = span_attributes(call, outcome, platform, index, len(calls))
        span = tracer.start_span(
            SPAN_NAME, context=context, start_time=started_ns, attributes=attributes
        )
        error = attributes.get("tool.error")
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            failed += 1
        span.end(end_time=ended_ns)
        spans.append(_readable(span))
    if failed:
        request.set_status(Status(StatusCode.ERROR, f"{failed} of {len(calls)} tool calls failed"))
    request.end(end_time=ended_ns)
    spans.append(_readable(request))
    return spans


def export_spans(exporter: SpanExporter, spans: Sequence[ReadableSpan]) -> SpanExportResult:
    """One POST for the batch, bounded by the exporter's timeout. Never raises."""
    try:
        return exporter.export(spans)
    except Exception:
        logger.warning("mock_tool_spans_export_raised", exc_info=True)
        return SpanExportResult.FAILURE
    finally:
        try:
            exporter.shutdown()
        except Exception:
            logger.warning("mock_tool_spans_shutdown_raised", exc_info=True)


async def _export(
    api_key: str,
    api_base: str,
    resource: Callable[[], Resource],
    simulation_id: str,
    platform: str,
    calls: Sequence[ToolCall],
    outcomes: Sequence[Outcome],
    started_ns: int,
    ended_ns: int,
) -> None:
    try:
        async with _exports:
            spans = build_spans(resource(), platform, calls, outcomes, started_ns, ended_ns)
            exporter = coval_exporter(api_key, simulation_id, api_base)
            result = await asyncio.to_thread(export_spans, exporter, spans)
    except Exception:
        logger.warning(
            "mock_tool_spans_failed", simulation_id=simulation_id, platform=platform, exc_info=True
        )
        return
    if result is SpanExportResult.SUCCESS:
        logger.info(
            "mock_tool_spans_exported",
            simulation_id=simulation_id,
            platform=platform,
            count=len(calls),
        )
    else:
        logger.warning(
            "mock_tool_spans_failed",
            simulation_id=simulation_id,
            platform=platform,
            count=len(calls),
        )


def schedule_export(
    *,
    api_key: SecretStr | None,
    api_base: str,
    resource: Callable[[], Resource],
    simulation_id: str,
    platform: str,
    calls: Sequence[ToolCall],
    outcomes: Sequence[Outcome],
    started_ns: int,
    ended_ns: int,
) -> bool:
    """Queue Coval's copy off the request path; returns whether anything was queued.

    With no key configured the appliance is simply not tracing, which is worth
    one warning per process, not one per tool call.
    """
    global _disabled_warned
    if api_key is None:
        if not _disabled_warned:
            logger.warning("mock_tool_traces_disabled", reason="no coval api key configured")
            _disabled_warned = True
        return False
    task = asyncio.get_running_loop().create_task(
        _export(
            api_key.get_secret_value(),
            api_base,
            resource,
            simulation_id,
            platform,
            calls,
            outcomes,
            started_ns,
            ended_ns,
        )
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True
