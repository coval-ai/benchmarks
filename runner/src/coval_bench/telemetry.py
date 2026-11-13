# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry counters for the runner.

Call :func:`configure_metrics` once at process startup and
:func:`shutdown_metrics` before exit. When nothing is configured the
``record_*`` helpers return immediately.

* ``benchmark_items{kind, provider, model, result}`` — one increment per
  dataset item a provider was asked to run; ``result`` is ``ok`` or ``error``.
* ``benchmark_runs{kind, job, status}`` — one increment per finished run.
* ``probe_results{kind, provider, model, outcome}`` — one increment per latency
  probe stored.
* ``sweep_items{kind, provider, model, outcome}`` — one increment per sweep item
  that got an outcome.
* ``sweep_queued{kind}`` — the items a sweep run left queued with tries to spare.

``outcome`` is ``ok``, ``provider`` for a failure that's the provider's, or what
went wrong on our side: ``auth``, ``credit``, ``rate_limit``, or ``ours`` for
anything else.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog
from opentelemetry.metrics import Counter as OtelCounter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource

if TYPE_CHECKING:
    from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.telemetry")

SERVICE_NAME = "benchmarks-runner"
ITEMS_COUNTER = "benchmark_items"
RUNS_COUNTER = "benchmark_runs"
PROBES_COUNTER = "probe_results"
SWEEP_ITEMS_COUNTER = "sweep_items"
SWEEP_QUEUED_GAUGE = "sweep_queued"

_provider: MeterProvider | None = None
_items: OtelCounter | None = None
_runs: OtelCounter | None = None
_probes: OtelCounter | None = None
_sweep_items: OtelCounter | None = None
_sweep_queued: Any | None = None


def build_resource(settings: Settings, service_name: str = SERVICE_NAME) -> Resource:
    """Who is reporting: shared by the metrics pipeline and the mock-tool traces."""
    instance = os.environ.get("CLOUD_RUN_EXECUTION") or socket.gethostname()
    return Resource.create(
        {
            "service.name": service_name,
            "service.instance.id": instance,
            "deployment.environment": settings.otel_deployment_environment,
        }
    )


def _build_reader(exporter_name: str) -> MetricReader:
    if exporter_name == "gcp":
        from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter

        return PeriodicExportingMetricReader(CloudMonitoringMetricsExporter())
    if exporter_name == "console":
        return PeriodicExportingMetricReader(ConsoleMetricExporter())
    raise ValueError(f"unknown otel_metrics_exporter {exporter_name!r}")


def configure_metrics(settings: Settings, *, readers: Sequence[MetricReader] | None = None) -> None:
    """Build the meter provider and counters; *readers* overrides the configured exporter."""
    global _provider, _items, _runs, _probes, _sweep_items, _sweep_queued
    shutdown_metrics()

    if readers is None:
        if settings.otel_metrics_exporter == "none":
            return
        try:
            readers = [_build_reader(settings.otel_metrics_exporter)]
        except Exception:
            logger.warning(
                "otel_metrics_exporter_init_failed",
                exporter=settings.otel_metrics_exporter,
                exc_info=True,
            )
            return

    _provider = MeterProvider(resource=build_resource(settings), metric_readers=list(readers))
    meter = _provider.get_meter("coval_bench")
    _items = meter.create_counter(
        ITEMS_COUNTER, unit="1", description="Dataset items run per provider, model and result"
    )
    _runs = meter.create_counter(RUNS_COUNTER, unit="1", description="Finished benchmark runs")
    _probes = meter.create_counter(
        PROBES_COUNTER, unit="1", description="Latency probes stored, by outcome"
    )
    _sweep_items = meter.create_counter(
        SWEEP_ITEMS_COUNTER, unit="1", description="Sweep items that got an outcome"
    )
    _sweep_queued = meter.create_gauge(
        SWEEP_QUEUED_GAUGE, unit="1", description="Items a sweep run left queued"
    )


def shutdown_metrics() -> None:
    """Flush and release the provider. Safe to call when nothing is configured."""
    global _provider, _items, _runs, _probes, _sweep_items, _sweep_queued
    provider, _provider, _items, _runs = _provider, None, None, None
    _probes = _sweep_items = _sweep_queued = None
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception:
        logger.warning("otel_metrics_shutdown_failed", exc_info=True)


def item_result(results: list[Any], result_status: Any) -> str | None:  # noqa: ANN401 — ResultStatus enum
    """``ok`` when at least one row succeeded, ``error`` otherwise, ``None`` for no rows."""
    if not results:
        return None
    ok = any(r.status is result_status.SUCCESS for r in results)
    return "ok" if ok else "error"


def record_item(*, kind: str, provider: str, model: str, result: str) -> None:
    """Count one dataset item for a provider and model."""
    if _items is None:
        return
    try:
        _items.add(
            1,
            {
                "kind": kind.lower(),
                "provider": provider.lower(),
                "model": model.lower(),
                "result": result,
            },
        )
    except Exception:
        logger.warning("otel_record_item_failed", exc_info=True)


def record_run(*, kind: str, status: str) -> None:
    """Count one finished run; ``job`` is the Cloud Run job name, ``local`` elsewhere."""
    if _runs is None:
        return
    job = os.environ.get("CLOUD_RUN_JOB", "local")
    try:
        _runs.add(1, {"kind": kind.lower(), "job": job.lower(), "status": status.lower()})
    except Exception:
        logger.warning("otel_record_run_failed", exc_info=True)


def outcome(error: str | None, *, ours: bool = False) -> str:
    """The ``outcome`` label for a call: ok, a key failure, or whose fault it otherwise was."""
    from coval_bench.arena.provider_health import classify_failure

    if error is None:
        return "ok"
    reason = classify_failure(None, error)
    if reason is not None:
        return reason.value
    return "ours" if ours else "provider"


def record_probe(*, kind: str, provider: str, model: str, outcome: str) -> None:
    """Count one stored latency probe."""
    _count(_probes, kind=kind, provider=provider, model=model, outcome=outcome)


def record_sweep_item(*, kind: str, provider: str, model: str, outcome: str) -> None:
    """Count one sweep item that got an outcome."""
    _count(_sweep_items, kind=kind, provider=provider, model=model, outcome=outcome)


def record_sweep_queued(*, kind: str, queued: int) -> None:
    """Report how many items a sweep run left queued with tries to spare."""
    if _sweep_queued is None:
        return
    try:
        _sweep_queued.set(queued, {"kind": kind.lower()})
    except Exception:
        logger.warning("otel_record_sweep_queued_failed", exc_info=True)


def _count(counter: OtelCounter | None, *, outcome: str, **labels: str) -> None:
    if counter is None:
        return
    try:
        counter.add(1, {**{k: v.lower() for k, v in labels.items()}, "outcome": outcome})
    except Exception:
        logger.warning("otel_record_failed", exc_info=True)
