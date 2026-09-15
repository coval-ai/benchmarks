# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coval_bench.telemetry."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, NumberDataPoint

from coval_bench import telemetry
from coval_bench.config import Settings
from coval_bench.db.models import Benchmark, Result, ResultStatus


@pytest.fixture(autouse=True)
def _reset_metrics() -> Iterator[None]:
    yield
    telemetry.shutdown_metrics()


def _settings(**overrides: Any) -> Settings:
    return Settings(database_url="postgresql://u:p@localhost:5432/x", **overrides)


Labels = tuple[tuple[str, str], ...]


def counter_points(reader: InMemoryMetricReader) -> dict[str, dict[Labels, int]]:
    """``{counter name: {sorted attribute pairs: value}}`` for every exported point."""
    data = reader.get_metrics_data()
    out: dict[str, dict[Labels, int]] = {}
    if data is None:
        return out
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for point in metric.data.data_points:
                    assert isinstance(point, NumberDataPoint)
                    labels = tuple(sorted((k, str(v)) for k, v in (point.attributes or {}).items()))
                    out.setdefault(metric.name, {})[labels] = int(point.value)
    return out


def sums(reader: InMemoryMetricReader, name: str) -> dict[Labels, int]:
    return counter_points(reader).get(name, {})


def _row(provider: str, model: str, metric: str, status: ResultStatus) -> Result:
    return Result(
        run_id=1,
        provider=provider,
        model=model,
        benchmark=Benchmark.STT,
        metric_type=metric,
        metric_value=1.0 if status is ResultStatus.SUCCESS else None,
        metric_units=None,
        status=status,
        error=None if status is ResultStatus.SUCCESS else "boom",
    )


def test_unconfigured_recording_is_a_noop() -> None:
    telemetry.configure_metrics(_settings())
    telemetry.record_item(kind="STT", provider="deepgram", model="nova-3", result="ok")
    telemetry.record_run(kind="stt", status="succeeded")


def test_counters_carry_lowercase_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD_RUN_JOB", "Benchmarks-Runner")
    reader = InMemoryMetricReader()
    telemetry.configure_metrics(_settings(), readers=[reader])

    telemetry.record_item(kind="STT", provider="Deepgram", model="Nova-3", result="ok")
    telemetry.record_item(kind="STT", provider="Deepgram", model="Nova-3", result="ok")
    telemetry.record_item(kind="TTS", provider="rime", model="arcana", result="error")
    telemetry.record_run(kind="both", status="PARTIAL")

    items = sums(reader, telemetry.ITEMS_COUNTER)
    assert (
        items[(("kind", "stt"), ("model", "nova-3"), ("provider", "deepgram"), ("result", "ok"))]
        == 2
    )
    assert (
        items[(("kind", "tts"), ("model", "arcana"), ("provider", "rime"), ("result", "error"))]
        == 1
    )
    runs = sums(reader, telemetry.RUNS_COUNTER)
    assert runs == {(("job", "benchmarks-runner"), ("kind", "both"), ("status", "partial")): 1}


def test_run_job_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUD_RUN_JOB", raising=False)
    reader = InMemoryMetricReader()
    telemetry.configure_metrics(_settings(), readers=[reader])
    telemetry.record_run(kind="stt", status="failed")
    assert list(sums(reader, telemetry.RUNS_COUNTER)) == [
        (("job", "local"), ("kind", "stt"), ("status", "failed"))
    ]


def test_resource_names_the_service() -> None:
    reader = InMemoryMetricReader()
    telemetry.configure_metrics(_settings(otel_deployment_environment="staging"), readers=[reader])
    telemetry.record_run(kind="stt", status="succeeded")
    data = reader.get_metrics_data()
    assert data is not None
    attrs = data.resource_metrics[0].resource.attributes
    assert attrs["service.name"] == telemetry.SERVICE_NAME
    assert attrs["deployment.environment"] == "staging"
    assert attrs["service.instance.id"]


def test_item_result_needs_one_success() -> None:
    ok = _row("deepgram", "nova-3", "ttft", ResultStatus.SUCCESS)
    bad = _row("deepgram", "nova-3", "wer", ResultStatus.FAILED)
    assert telemetry.item_result([ok, bad], ResultStatus) == "ok"
    assert telemetry.item_result([bad, bad], ResultStatus) == "error"
    assert telemetry.item_result([], ResultStatus) is None


def test_shutdown_is_idempotent() -> None:
    telemetry.shutdown_metrics()
    telemetry.configure_metrics(_settings(), readers=[InMemoryMetricReader()])
    telemetry.shutdown_metrics()
    telemetry.shutdown_metrics()
    telemetry.record_run(kind="stt", status="succeeded")


def test_console_exporter_configures() -> None:
    telemetry.configure_metrics(_settings(otel_metrics_exporter="console"))
    assert telemetry._provider is not None
