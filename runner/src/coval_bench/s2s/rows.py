# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Coval metric values to Result rows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

import structlog

from coval_bench.db.models import Result, ResultStatus
from coval_bench.db.writer import RunWriter
from coval_bench.registries import METRIC_SPECS, Metric
from coval_bench.s2s.conditions import DatasetMetrics
from coval_bench.s2s.specs import AgentSpec

logger = structlog.get_logger("coval_bench.s2s.rows")


def v2v_value(raw: object) -> tuple[float | None, ResultStatus] | None:
    """Seconds to ms; a clip with no numeric value becomes a FAILED row."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return round(float(raw) * 1000, 1), ResultStatus.SUCCESS
    return None, ResultStatus.FAILED


class InvalidInstructionVerdict(Exception):
    """A run's instruction metric returned a value outside YES/NO/UNKNOWN."""


def instruction_verdict(raw: object) -> bool | None:
    """Classify a binary judge verdict: YES -> True, NO -> False, UNKNOWN -> None.

    UNKNOWN (None) is excluded from the pass rate. A binary judge only ever emits
    canonical YES/NO/UNKNOWN, so anything else violates the contract and raises,
    rather than being silently scored as a miss.
    """
    normalized = raw.strip().upper() if isinstance(raw, str) else raw
    if normalized == "YES":
        return True
    if normalized == "NO":
        return False
    if normalized == "UNKNOWN":
        return None
    raise InvalidInstructionVerdict(f"unexpected instruction verdict: {raw!r}")


def has_duplicate_ids(values: list[dict[str, Any]]) -> bool:
    """True if two of a metric's conversations share a simulation id."""
    ids = [v.get("simulation_output_id") for v in values]
    return len(ids) != len(set(ids))


def population_mismatch(
    anchor_values: list[dict[str, Any]], other_values: list[dict[str, Any]]
) -> dict[str, object] | None:
    """None if *other_values* covers the same conversations as the anchor.

    Ids, not counts: equal counts can be different conversations. Values with no id
    are excluded from both sides — two of them are not the same conversation, so
    matching them would invent an overlap. The caller trims to the overlap; only
    duplicates skip the metric for the run.
    """
    other_set = {sid for v in other_values if (sid := v.get("simulation_output_id"))}
    anchor_set = {sid for v in anchor_values if (sid := v.get("simulation_output_id"))}
    missing = sorted(str(x) for x in anchor_set - other_set)
    extra = sorted(str(x) for x in other_set - anchor_set)
    duplicate = has_duplicate_ids(other_values)
    if not duplicate and not missing and not extra:
        return None
    return {"missing_ids": missing, "extra_ids": extra, "duplicate_ids": duplicate}


def instruction_value(raw: object) -> tuple[float | None, ResultStatus] | None:
    """YES -> 100.0, NO -> 0.0, UNKNOWN -> no row, so the mean is YES / (YES + NO).

    A composite judge reports the fraction of expected behaviors met instead of a
    verdict; that fraction becomes a percentage on the same scale, unrounded.
    Raises InvalidInstructionVerdict on any value outside either contract.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not 0.0 <= raw <= 1.0:
            raise InvalidInstructionVerdict(f"instruction fraction out of range: {raw!r}")
        return float(raw) * 100.0, ResultStatus.SUCCESS
    verdict = instruction_verdict(raw)
    if verdict is None:
        return None
    return (100.0 if verdict else 0.0), ResultStatus.SUCCESS


def interruption_value(raw: object) -> tuple[float | None, ResultStatus] | None:
    """Interruptions per minute, unrounded; a clip with no numeric value becomes a FAILED row."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), ResultStatus.SUCCESS
    return None, ResultStatus.FAILED


def call_length_value(raw: object) -> tuple[float | None, ResultStatus] | None:
    """Conversation audio duration in seconds; a clip with no numeric value becomes a FAILED row."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), ResultStatus.SUCCESS
    return None, ResultStatus.FAILED


# The only per-metric part: one raw Coval value -> (row value, status), or None to
# write no row. A metric is ingestable once it appears here.
VALUE_MAPPERS: dict[Metric, Callable[[object], tuple[float | None, ResultStatus] | None]] = {
    Metric.V2V: v2v_value,
    Metric.INSTRUCTION_FOLLOWING: instruction_value,
    Metric.INTERRUPTION_RATE: interruption_value,
    Metric.CALL_LENGTH: call_length_value,
    Metric.WORKFLOW_ADHERENCE: instruction_value,
}


def s2s_rows(
    values: list[dict[str, Any]],
    *,
    metric: Metric,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """Map one metric's per-conversation values to Result rows."""
    to_row = VALUE_MAPPERS[metric]
    rows: list[Result] = []
    for i, v in enumerate(values):
        mapped = to_row(v.get("value"))
        if mapped is None:
            continue
        metric_value, status = mapped
        sim_id = v.get("simulation_output_id")
        rows.append(
            Result(
                run_id=run_pk,
                provider=spec.provider,
                model=spec.model,
                benchmark=spec.benchmark,
                metric_type=metric,
                metric_units=METRIC_SPECS[metric].units,
                metric_value=metric_value,
                audio_filename=f"{coval_run_id}/{sim_id}" if sim_id else f"{coval_run_id}/{i}",
                status=status,
            )
        )
    return rows


def failed_conversation_rows(
    output_ids: list[str],
    anchor_values: list[dict[str, Any]],
    *,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """A FAILED latency row for each conversation with no anchor value.

    Coval omits a failed conversation from every metric's values but still lists
    it in ``results.output_ids``. Without these rows a model is judged only on
    the calls it survived, and a run's failures would not show in its success
    rate.

    An anchor value without a simulation_output_id (kept by ``s2s_rows`` under
    its index fallback) can't be reconciled against ``output_ids``, so coverage
    is undecidable for the whole run: synthesize nothing rather than stamp a
    measured conversation FAILED.
    """
    covered = {v.get("simulation_output_id") for v in anchor_values}
    if not all(covered):
        logger.warning(
            "anchor_ids_missing",
            provider=spec.provider,
            coval_run_id=coval_run_id,
            anchors_without_id=sum(1 for v in anchor_values if not v.get("simulation_output_id")),
        )
        return []
    return [
        Result(
            run_id=run_pk,
            provider=spec.provider,
            model=spec.model,
            benchmark=spec.benchmark,
            metric_type=Metric.V2V,
            metric_units=METRIC_SPECS[Metric.V2V].units,
            metric_value=None,
            audio_filename=f"{coval_run_id}/{sim_id}",
            status=ResultStatus.FAILED,
        )
        for sim_id in output_ids
        if sim_id not in covered
    ]


# Metrics a condition declares ``local`` are measured on our side and looked up
# by the anchor's conversation ids instead of fetched from Coval. A local metric
# is ingestable once it appears here, mirroring VALUE_MAPPERS.
LOCAL_SOURCES: dict[
    Metric, Callable[[RunWriter, Sequence[str]], Awaitable[Mapping[str, float]]]
] = {
    Metric.TTFT: lambda writer, sim_ids: writer.conversation_ttft(sim_ids),
}


def local_rows(
    values: Mapping[str, float],
    *,
    metric: Metric,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """One SUCCESS row per conversation measured locally, keyed like the anchor rows."""
    return [
        Result(
            run_id=run_pk,
            provider=spec.provider,
            model=spec.model,
            benchmark=spec.benchmark,
            metric_type=metric,
            metric_units=METRIC_SPECS[metric].units,
            metric_value=round(value, 3),
            audio_filename=f"{coval_run_id}/{sim_id}",
            status=ResultStatus.SUCCESS,
        )
        for sim_id, value in values.items()
    ]


def metric_values(metrics: dict[str, Any], metric_id: str | None) -> list[dict[str, Any]] | None:
    """The metric's per-conversation values, or None when it is not on the run."""
    payload = metrics.get(metric_id) if metric_id else None
    if payload is None:
        return None
    return cast("list[dict[str, Any]]", payload.get("values", []))


def ingestable_metrics(
    condition: DatasetMetrics, metric_ids: Mapping[Metric, str]
) -> frozenset[Metric]:
    """The condition's metrics that are configured and have a row builder."""
    fetched = frozenset(
        metric for metric in condition.fetched if metric in metric_ids and metric in VALUE_MAPPERS
    )
    return fetched | (condition.local & frozenset(LOCAL_SOURCES))
