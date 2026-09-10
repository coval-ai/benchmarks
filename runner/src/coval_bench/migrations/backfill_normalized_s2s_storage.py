# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608  # Audited SQL is intentionally kept as complete statements.
"""Backfill legacy S2S rows into normalized database-only storage.

Historic S2S results retain metric outcomes, but not the underlying conversation
artifact.  This standalone operator therefore writes observations, evaluations,
values, and rollups only.  It never fabricates artifact lineage and never changes
legacy rows.
"""

from __future__ import annotations

import json
import math
import signal
import sys
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from types import FrameType
from typing import Any, TextIO

import click
import psycopg

from coval_bench.config import get_settings
from coval_bench.s2s.fetch_v2v import _normalized_dataset_sha256

_OBS_NAMESPACE = uuid.UUID("0c43bb05-d3e4-5d07-b9d9-1e4e7e18f9f2")
_EVAL_NAMESPACE = uuid.UUID("5bb93c92-caae-5f2b-a7c5-3a239e8d423d")
_LOCK = "normalized_storage_backfill"
_FALLBACK_OBSERVATION_ERROR = "Coval conversation produced no successful metric value"
_FALLBACK_EVALUATION_ERROR = "legacy metric produced no value"
_MISMATCH_DETAIL_LIMIT = 100
_PROGRESS_INTERVAL_SECONDS = 30.0
_PROGRESS_MAX_UNITS = 10
_PUBLIC_PARITY_STATEMENT_TIMEOUT = "30s"
_METRICS = {
    "V2V": "milliseconds",
    "InstructionFollowing": "percent",
    "InterruptionRate": "per_minute",
}
_MISMATCH_KEYS = {
    "payload_mismatches": "payload_mismatch_count",
    "observation_population_mismatches": "observation_population_mismatch_count",
    "public_parity_mismatches": "public_parity_mismatch_count",
    "rollup_mismatches": "rollup_mismatch_count",
}


class _Cancelled(Exception):
    """Raised by the CLI SIGTERM handler so cleanup finally blocks run."""


@contextmanager
def _sigterm() -> Iterator[None]:
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(_: int, __: FrameType | None) -> None:
        raise _Cancelled()

    signal.signal(signal.SIGTERM, cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@dataclass(frozen=True)
class LegacyS2SRow:
    id: int
    run_id: int
    dataset_id: str
    dataset_sha256: str
    scheduled_at: datetime | None
    provider: str
    model: str
    voice: str | None
    metric: str
    value: float | None
    unit: str | None
    sample_id: str | None
    status: str
    error: str | None
    captured_at: datetime


@dataclass(frozen=True)
class Plan:
    rows: tuple[LegacyS2SRow, ...]
    status: str
    error: str | None

    @property
    def first(self) -> LegacyS2SRow:
        return self.rows[0]

    @property
    def natural(self) -> tuple[Any, ...]:
        row = self.first
        return row.run_id, row.sample_id, row.provider, row.model, row.voice

    @property
    def id(self) -> uuid.UUID:
        return uuid.uuid5(_OBS_NAMESPACE, "|".join(map(str, self.natural)))

    def evaluation_id(self, metric: str) -> uuid.UUID:
        return uuid.uuid5(_EVAL_NAMESPACE, f"{self.id}|{metric}|v1|default")

    @property
    def dataset_sha256(self) -> str:
        return _normalized_dataset_sha256(self.first.dataset_sha256)

    @property
    def population_key(self) -> tuple[Any, ...]:
        row = self.first
        return (*self.natural, row.dataset_id)

    def evaluations(self) -> list[tuple[str, str, float | None, str | None]]:
        evaluations: list[tuple[str, str, float | None, str | None]] = []
        for row in sorted(self.rows, key=lambda candidate: candidate.metric):
            succeeded = row.status == "success" and row.value is not None
            legacy_error = row.error if row.error else None
            evaluations.append(
                (
                    row.metric,
                    "succeeded" if succeeded else "failed",
                    float(row.value) if row.status == "success" and row.value is not None else None,
                    None if succeeded else (legacy_error or _FALLBACK_EVALUATION_ERROR),
                )
            )
        return evaluations


@dataclass(frozen=True)
class ReconcileResult:
    disposition: str
    reason: str | None = None
    evaluations: int = 0
    values: int = 0


@dataclass
class PageDelta:
    eligible: int = 0
    created: int = 0
    reconciled: int = 0
    live_owned: int = 0
    evaluations: int = 0
    values: int = 0
    buckets: int = 0
    conflicts: Counter[str] = field(default_factory=Counter)

    def record(self, result: ReconcileResult) -> None:
        if result.disposition == "eligible":
            self.eligible += 1
        elif result.disposition == "created":
            self.eligible += 1
            self.created += 1
            self.evaluations += result.evaluations
            self.values += result.values
        elif result.disposition == "reconciled":
            self.reconciled += 1
        elif result.disposition == "live_owned":
            self.reconciled += 1
            self.live_owned += 1
        else:
            self.conflicts[result.reason or "stored_payload_conflict"] += 1


@dataclass
class ProgressReporter:
    """Write aggregate-only, machine-readable progress events to stderr."""

    mode: str
    min_result_id: int
    max_result_id: int
    batch_size: int
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    monotonic: Callable[[], float] = time.monotonic
    interval_seconds: float = _PROGRESS_INTERVAL_SECONDS
    max_units: int = _PROGRESS_MAX_UNITS
    started_at: float = field(init=False)
    last_emitted_at: float = field(init=False)
    phase_started_at: float = field(init=False)
    units_since_emit: int = 0
    pages: int = 0
    runs: int = 0
    results: int = 0
    conversations: int = 0
    phase_pages: int = 0
    phase_runs: int = 0
    phase_results: int = 0
    phase_conversations: int = 0
    phase_units_completed: int = 0
    total_runs: int | None = None
    phase_total_runs: int | None = None
    last_completed_run_id: int | None = None
    last_completed_result_id: int | None = None
    last_committed_run_id: int | None = None
    last_committed_result_id: int | None = None
    phase_last_completed_run_id: int | None = None
    phase_last_completed_result_id: int | None = None

    def __post_init__(self) -> None:
        now = self.monotonic()
        self.started_at = now
        self.last_emitted_at = now
        self.phase_started_at = now

    def _payload(self, phase: str, status: str, report: Mapping[str, Any]) -> dict[str, Any]:
        now = self.monotonic()
        elapsed = max(now - self.started_at, 0.0)
        phase_elapsed = max(now - self.phase_started_at, 0.0)
        phase_run_rate = self.phase_runs / phase_elapsed if phase_elapsed else 0.0
        mismatch_count = sum(int(report.get(key, 0)) for key in _MISMATCH_KEYS.values())
        skipped = _counter_total(report.get("skipped_by_reason")) + _counter_total(
            report.get("verification_skipped_by_reason")
        )
        conflicts = _counter_total(report.get("conflicts_by_reason")) + _counter_total(
            report.get("verification_conflicts_by_reason")
        )
        payload: dict[str, Any] = {
            "event": "normalized_s2s_storage_backfill_progress",
            "phase": phase,
            "status": status,
            "mode": self.mode,
            "min_result_id": self.min_result_id,
            "max_result_id": self.max_result_id,
            "batch_size": self.batch_size,
            "pages": self.pages,
            "runs": self.runs,
            "results": self.results,
            "conversations": self.conversations,
            "phase_pages": self.phase_pages,
            "phase_runs": self.phase_runs,
            "phase_results": self.phase_results,
            "phase_conversations": self.phase_conversations,
            "phase_units_completed": self.phase_units_completed,
            "total_runs": self.total_runs,
            "phase_total_runs": self.phase_total_runs,
            "last_completed_run_id": self.last_completed_run_id,
            "last_completed_result_id": self.last_completed_result_id,
            "last_committed_run_id": self.last_committed_run_id,
            "last_committed_result_id": self.last_committed_result_id,
            "phase_last_completed_run_id": self.phase_last_completed_run_id,
            "phase_last_completed_result_id": self.phase_last_completed_result_id,
            "elapsed_seconds": round(elapsed, 3),
            "phase_elapsed_seconds": round(phase_elapsed, 3),
            "throughput_runs_per_second": round(self.runs / elapsed, 6) if elapsed else 0.0,
            "throughput_results_per_second": round(self.results / elapsed, 6) if elapsed else 0.0,
            "throughput_conversations_per_second": round(self.conversations / elapsed, 6)
            if elapsed
            else 0.0,
            "phase_throughput_runs_per_second": round(phase_run_rate, 6),
            "phase_throughput_results_per_second": round(self.phase_results / phase_elapsed, 6)
            if phase_elapsed
            else 0.0,
            "phase_throughput_conversations_per_second": round(
                self.phase_conversations / phase_elapsed, 6
            )
            if phase_elapsed
            else 0.0,
            "skipped": skipped,
            "conflicts": conflicts,
            "mismatches": mismatch_count,
        }
        payload["eta_seconds"] = (
            round(max(self.phase_total_runs - self.phase_runs, 0) / phase_run_rate, 3)
            if self.phase_total_runs is not None and phase_run_rate
            else None
        )
        return payload

    def emit(self, phase: str, status: str, report: Mapping[str, Any]) -> None:
        self.stream.write(json.dumps(self._payload(phase, status, report), sort_keys=True) + "\n")
        self.stream.flush()
        self.last_emitted_at = self.monotonic()
        self.units_since_emit = 0

    def phase_started(
        self, phase: str, report: Mapping[str, Any], *, total_runs: int | None = None
    ) -> None:
        self.phase_started_at = self.monotonic()
        self.phase_pages = 0
        self.phase_runs = 0
        self.phase_results = 0
        self.phase_conversations = 0
        self.phase_units_completed = 0
        self.phase_last_completed_run_id = None
        self.phase_last_completed_result_id = None
        self.phase_total_runs = total_runs
        self.emit(phase, "started", report)

    def phase_completed(self, phase: str, report: Mapping[str, Any]) -> None:
        self.emit(phase, "completed", report)

    def completed_page(
        self,
        run_ids: list[int],
        rows: list[LegacyS2SRow],
        conversations: int,
        report: Mapping[str, Any],
        *,
        phase: str,
        committed: bool = False,
    ) -> None:
        self.pages += 1
        self.runs += len(run_ids)
        self.results += len(rows)
        self.conversations += conversations
        self.phase_pages += 1
        self.phase_runs += len(run_ids)
        self.phase_results += len(rows)
        self.phase_conversations += conversations
        self.last_completed_run_id = run_ids[-1]
        self.last_completed_result_id = max((row.id for row in rows), default=None)
        self.phase_last_completed_run_id = self.last_completed_run_id
        self.phase_last_completed_result_id = self.last_completed_result_id
        if committed:
            self.last_committed_run_id = self.last_completed_run_id
            self.last_committed_result_id = self.last_completed_result_id
        self.completed_unit(report, phase=phase)

    def completed_unit(self, report: Mapping[str, Any], *, phase: str) -> None:
        self.units_since_emit += 1
        self.phase_units_completed += 1
        if (
            self.units_since_emit >= self.max_units
            or self.monotonic() - self.last_emitted_at >= self.interval_seconds
        ):
            self.emit(phase, "progress", report)


def _counter_total(value: object) -> int:
    return sum(int(count) for count in value.values()) if isinstance(value, Mapping) else 0


def _report(low: int, high: int, batch: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "window": {"min_result_id": low, "max_result_id": high, "batch_size": batch},
        "source_rows": 0,
        "source_runs": 0,
        "source_groups": 0,
        "source_pages": 0,
        "eligible": 0,
        "created": 0,
        "reconciled": 0,
        "live_owned": 0,
        "evaluations": 0,
        "values": 0,
        "buckets": 0,
        "skipped_by_reason": Counter(),
        "conflicts_by_reason": Counter(),
        "verification_rows": 0,
        "verification_runs": 0,
        "verification_groups": 0,
        "verification_pages": 0,
        "verification_reconciled": 0,
        "verification_live_owned": 0,
        "verification_skipped_by_reason": Counter(),
        "verification_conflicts_by_reason": Counter(),
        "backfill_complete": False,
    }
    for details_key, count_key in _MISMATCH_KEYS.items():
        report[details_key] = []
        report[count_key] = 0
        report[f"{details_key}_truncated"] = False
    return report


def _append_mismatch(
    report: dict[str, Any], key: str, detail: dict[str, Any], *, count: int = 1
) -> None:
    count_key = _MISMATCH_KEYS[key]
    report[count_key] += count
    if len(report[key]) < _MISMATCH_DETAIL_LIMIT:
        report[key].append(detail)
    else:
        report[f"{key}_truncated"] = True


def _record_counter_diff(
    report: dict[str, Any], key: str, expected: Counter[Any], actual: Counter[Any]
) -> None:
    for item in sorted(set(expected) | set(actual), key=repr):
        expected_count = expected[item]
        actual_count = actual[item]
        if expected_count == actual_count:
            continue
        _append_mismatch(
            report,
            key,
            {
                "identity": list(item),
                "expected_count": expected_count,
                "actual_count": actual_count,
            },
            count=abs(expected_count - actual_count),
        )


_PAGE_SQL = """
SELECT DISTINCT run_id
FROM benchmarks_v2.results
WHERE id BETWEEN %s AND %s AND benchmark='S2S' AND run_id>%s
ORDER BY run_id LIMIT %s
"""
_ROWS_SQL = """
SELECT r.id,r.run_id,n.dataset_id,n.dataset_sha256,n.scheduled_at,
       r.provider,r.model,r.voice,r.metric_type,r.metric_value,r.metric_units,
       r.audio_filename,r.status,r.error,r.created_at
FROM benchmarks_v2.results r
JOIN benchmarks_v2.runs n ON n.id=r.run_id
WHERE r.run_id=ANY(%s) AND r.id BETWEEN %s AND %s AND r.benchmark='S2S'
ORDER BY r.run_id,r.id
"""


def _read_complete_page(
    conn: psycopg.Connection,
    low: int,
    high: int,
    after: int,
    size: int,
    skipped: Counter[str],
) -> tuple[list[int], list[LegacyS2SRow], list[LegacyS2SRow]]:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(_PAGE_SQL, (low, high, after, size))
        run_ids = [int(row[0]) for row in cur.fetchall()]
        if not run_ids:
            return [], [], []
        cur.execute(_ROWS_SQL, (run_ids, low, high))
        rows = [LegacyS2SRow(*row) for row in cur.fetchall()]
        cur.execute(
            """SELECT n.id,min(r.id),max(r.id),n.status
               FROM benchmarks_v2.runs n
               JOIN benchmarks_v2.results r ON r.run_id=n.id
               WHERE n.id=ANY(%s)
               GROUP BY n.id,n.status""",
            (run_ids,),
        )
        excluded: set[int] = set()
        for run_id, first_result, last_result, status in cur.fetchall():
            if first_result < low or last_result > high:
                skipped["split_window_run"] += 1
                excluded.add(int(run_id))
            elif status not in ("succeeded", "partial", "failed"):
                skipped["run_not_terminal"] += 1
                excluded.add(int(run_id))
        complete = [row for row in rows if row.run_id not in excluded]
        return run_ids, rows, complete


def _qualifying_run_count(conn: psycopg.Connection, low: int, high: int) -> int:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(
            "SELECT count(DISTINCT run_id) FROM benchmarks_v2.results WHERE id BETWEEN %s AND %s AND benchmark='S2S'",
            (low, high),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("qualifying run count returned no row")
        return int(row[0])


def _plans(rows: list[LegacyS2SRow], skipped: Counter[str], conflicts: Counter[str]) -> list[Plan]:
    grouped: dict[tuple[Any, ...], list[LegacyS2SRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.run_id, row.provider, row.model, row.voice, row.sample_id)].append(row)
    plans: list[Plan] = []
    for key, group in grouped.items():
        if not key[-1]:
            skipped["audio_filename_missing"] += 1
            continue
        shared = {
            (row.dataset_id, row.dataset_sha256, row.scheduled_at, row.captured_at) for row in group
        }
        metric_counts = Counter(row.metric for row in group)
        if len(shared) != 1 or any(
            metric not in _METRICS or count != 1 for metric, count in metric_counts.items()
        ):
            conflicts["conflicting_group"] += 1
            continue
        invalid = False
        for row in group:
            if row.status == "success":
                invalid = (
                    row.value is None
                    or not math.isfinite(float(row.value))
                    or row.unit != _METRICS[row.metric]
                )
            elif row.status != "failed" or row.value is not None:
                invalid = True
            if invalid:
                break
        if invalid:
            conflicts["metric_payload_invalid"] += 1
            continue
        succeeded = any(row.status == "success" and row.value is not None for row in group)
        legacy_error = next((row.error for row in group if row.error), None)
        plans.append(
            Plan(
                tuple(group),
                "succeeded" if succeeded else "failed",
                None if succeeded else (legacy_error or _FALLBACK_OBSERVATION_ERROR),
            )
        )
    return plans


def _no_lineage(cur: psycopg.Cursor[Any], observation_id: uuid.UUID) -> bool:
    for table in ("observation_artifacts", "preprocessing_artifacts"):
        cur.execute(
            f"SELECT EXISTS(SELECT 1 FROM benchmarks_v2.{table} WHERE observation_id=%s)",
            (observation_id,),
        )
        found = cur.fetchone()
        if found is None:
            raise RuntimeError("lineage existence query returned no row")
        if found[0]:
            return False
    cur.execute(
        """SELECT EXISTS(
               SELECT 1 FROM benchmarks_v2.metric_evaluation_inputs i
               JOIN benchmarks_v2.metric_evaluations e ON e.id=i.metric_evaluation_id
               WHERE e.observation_id=%s
           ) OR EXISTS(
               SELECT 1 FROM benchmarks_v2.metric_artifacts a
               JOIN benchmarks_v2.metric_evaluations e ON e.id=a.metric_evaluation_id
               WHERE e.observation_id=%s
           )""",
        (observation_id, observation_id),
    )
    found = cur.fetchone()
    if found is None:
        raise RuntimeError("metric lineage existence query returned no row")
    return not found[0]


def _match_reason(
    cur: psycopg.Cursor[Any], plan: Plan, observation_id: uuid.UUID, *, live_owned: bool
) -> str | None:
    row = plan.first
    cur.execute(
        """SELECT run_id,dataset_id,dataset_sha256,sample_id,provider,model,voice,
                  benchmark,source_kind,transport_protocol,submit_to_headers_ms,
                  provider_extras,captured_at,status,error,failure_origin
           FROM benchmarks_v2.benchmark_observations WHERE id=%s""",
        (observation_id,),
    )
    expected_observation = (
        row.run_id,
        row.dataset_id,
        plan.dataset_sha256,
        row.sample_id,
        row.provider,
        row.model,
        row.voice,
        "S2S",
        "conversation_audio",
        None,
        None,
        None,
        row.captured_at,
        plan.status,
        plan.error,
        None if plan.status == "succeeded" else "provider",
    )
    if cur.fetchone() != expected_observation:
        return "observation_payload_mismatch"
    if not _no_lineage(cur, observation_id):
        return "unexpected_artifact_lineage"
    cur.execute(
        """SELECT id,metric_type,metric_version,evaluation_variant,executor,
                  external_request_id,status,started_at,finished_at,error
           FROM benchmarks_v2.metric_evaluations
           WHERE observation_id=%s ORDER BY metric_type""",
        (observation_id,),
    )
    actual_evaluations = cur.fetchall()
    expected = {
        metric: (status, value, error) for metric, status, value, error in plan.evaluations()
    }
    if len(actual_evaluations) != len(expected):
        return "evaluation_population_mismatch"
    for (
        evaluation_id,
        metric,
        version,
        variant,
        executor,
        external_request_id,
        status,
        started_at,
        finished_at,
        error,
    ) in actual_evaluations:
        wanted = expected.get(metric)
        if wanted is None:
            return "unexpected_metric_evaluation"
        if not live_owned and evaluation_id != plan.evaluation_id(metric):
            return "backfill_evaluation_id_mismatch"
        if (version, variant, executor, external_request_id, status, error) != (
            "v1",
            "default",
            "coval_api",
            None,
            wanted[0],
            wanted[2],
        ):
            return "evaluation_payload_mismatch"
        if live_owned:
            if (
                started_at is None
                or finished_at is None
                or started_at < row.captured_at
                or finished_at < started_at
            ):
                return "live_evaluation_timestamps_invalid"
        elif started_at != row.captured_at or finished_at != row.captured_at:
            return "backfill_evaluation_timestamps_mismatch"
        cur.execute(
            """SELECT value_key,unit,value,value_role
               FROM benchmarks_v2.metric_values
               WHERE metric_evaluation_id=%s ORDER BY value_key""",
            (evaluation_id,),
        )
        values = cur.fetchall()
        expected_values = (
            [] if wanted[1] is None else [("primary", _METRICS[metric], wanted[1], "primary")]
        )
        if values != expected_values:
            return "metric_value_payload_mismatch"
    return None


def _insert_plan(cur: psycopg.Cursor[Any], plan: Plan) -> tuple[int, int]:
    row = plan.first
    cur.execute(
        """INSERT INTO benchmarks_v2.benchmark_observations
           (id,run_id,dataset_id,dataset_sha256,sample_id,provider,model,voice,
            benchmark,source_kind,captured_at,status,error,failure_origin)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'S2S','conversation_audio',%s,%s,%s,%s)""",
        (
            plan.id,
            row.run_id,
            row.dataset_id,
            plan.dataset_sha256,
            row.sample_id,
            row.provider,
            row.model,
            row.voice,
            row.captured_at,
            plan.status,
            plan.error,
            None if plan.status == "succeeded" else "provider",
        ),
    )
    evaluations = 0
    values = 0
    for metric, status, value, error in plan.evaluations():
        evaluation_id = plan.evaluation_id(metric)
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (id,observation_id,metric_type,metric_version,evaluation_variant,executor,status)
               VALUES (%s,%s,%s,'v1','default','coval_api','queued')""",
            (evaluation_id, plan.id, metric),
        )
        cur.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET status='running',started_at=%s WHERE id=%s",
            (row.captured_at, evaluation_id),
        )
        if value is not None:
            cur.execute(
                """INSERT INTO benchmarks_v2.metric_values
                   (metric_evaluation_id,value_key,unit,value,value_role)
                   VALUES (%s,'primary',%s,%s,'primary')""",
                (evaluation_id, _METRICS[metric], value),
            )
            values += 1
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status=%s,finished_at=%s,error=%s WHERE id=%s""",
            (status, row.captured_at, error, evaluation_id),
        )
        evaluations += 1
    return evaluations, values


def _reconcile(cur: psycopg.Cursor[Any], plan: Plan, *, apply: bool) -> ReconcileResult:
    cur.execute(
        """SELECT id FROM benchmarks_v2.benchmark_observations
           WHERE run_id=%s AND sample_id=%s AND provider=%s AND model=%s
             AND voice IS NOT DISTINCT FROM %s""",
        plan.natural,
    )
    found = cur.fetchone()
    if found is None:
        if not apply:
            return ReconcileResult("eligible")
        evaluations, values = _insert_plan(cur, plan)
        return ReconcileResult("created", evaluations=evaluations, values=values)
    observation_id = found[0]
    live_owned = observation_id != plan.id
    reason = _match_reason(cur, plan, observation_id, live_owned=live_owned)
    if reason is not None:
        return ReconcileResult("conflict", reason=reason)
    return ReconcileResult("live_owned" if live_owned else "reconciled")


def _merge_source_delta(report: dict[str, Any], delta: PageDelta) -> None:
    for key in (
        "eligible",
        "created",
        "reconciled",
        "live_owned",
        "evaluations",
        "values",
        "buckets",
    ):
        report[key] += getattr(delta, key)
    report["conflicts_by_reason"].update(delta.conflicts)


def _merge_verification_delta(report: dict[str, Any], delta: PageDelta) -> None:
    report["verification_reconciled"] += delta.reconciled
    report["verification_live_owned"] += delta.live_owned
    report["verification_conflicts_by_reason"].update(delta.conflicts)


_ROLLUP_PAYLOAD_SQL = """
SELECT o.provider,o.model,o.benchmark,COALESCE(o.dataset_id,'__all__'),
       e.metric_type,e.metric_version,e.evaluation_variant,v.value_key,v.unit,%(bucket)s,
       MIN(v.value)::float8,
       PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY v.value)::float8,
       PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY v.value)::float8,
       PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY v.value)::float8,
       MAX(v.value)::float8,SUM(v.value)::float8,COUNT(*)::int
FROM benchmarks_v2.metric_values v
JOIN benchmarks_v2.metric_evaluations e ON e.id=v.metric_evaluation_id
JOIN benchmarks_v2.benchmark_observations o ON o.id=e.observation_id
JOIN benchmarks_v2.runs r ON r.id=o.run_id
WHERE o.benchmark='S2S' AND o.status='succeeded' AND e.status='succeeded'
  AND r.status IN ('succeeded','partial') AND r.scheduled_at=%(bucket)s
GROUP BY GROUPING SETS (
  (o.provider,o.model,o.benchmark,o.dataset_id,e.metric_type,e.metric_version,
   e.evaluation_variant,v.value_key,v.unit),
  (o.provider,o.model,o.benchmark,e.metric_type,e.metric_version,
   e.evaluation_variant,v.value_key,v.unit)
)
"""
_STORED_ROLLUP_SQL = """
SELECT provider,model,benchmark,dataset_id,metric_type,metric_version,
       evaluation_variant,value_key,unit,bucket_at,min_value,p25,p50,p75,
       max_value,value_sum,sample_count
FROM benchmarks_v2.metric_values_by_bucket
WHERE bucket_at=%(bucket)s AND benchmark='S2S'
"""


def _refresh(cur: psycopg.Cursor[Any], bucket: datetime) -> None:
    params = {"bucket": bucket}
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended('metric_values_by_bucket',extract(epoch FROM %(bucket)s::timestamptz)::bigint))",
        params,
    )
    cur.execute(
        "DELETE FROM benchmarks_v2.metric_values_by_bucket "
        "WHERE bucket_at=%(bucket)s AND benchmark='S2S'",
        params,
    )
    cur.execute(
        """INSERT INTO benchmarks_v2.metric_values_by_bucket
           (provider,model,benchmark,dataset_id,metric_type,metric_version,evaluation_variant,
            value_key,unit,bucket_at,min_value,p25,p50,p75,max_value,value_sum,sample_count)
        """
        + _ROLLUP_PAYLOAD_SQL,
        params,
    )


_BUCKET_PAGE_SQL = """
SELECT n.scheduled_at
FROM benchmarks_v2.runs n
WHERE n.status IN ('succeeded','partial','failed') AND n.scheduled_at IS NOT NULL
  AND EXISTS (
    SELECT 1 FROM benchmarks_v2.results r
    WHERE r.run_id=n.id AND r.id BETWEEN %s AND %s AND r.benchmark='S2S'
  )
  AND NOT EXISTS (
    SELECT 1 FROM benchmarks_v2.results r
    WHERE r.run_id=n.id AND (r.id < %s OR r.id > %s)
  )
  AND (%s::timestamptz IS NULL OR n.scheduled_at > %s)
GROUP BY n.scheduled_at ORDER BY n.scheduled_at LIMIT %s
"""


def _scheduled_bucket_pages(
    conn: psycopg.Connection, low: int, high: int, batch_size: int
) -> Iterable[list[datetime]]:
    after: datetime | None = None
    while True:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(_BUCKET_PAGE_SQL, (low, high, low, high, after, after, batch_size))
            page = [row[0] for row in cur.fetchall()]
        if not page:
            return
        yield page
        after = page[-1]


def _verify_rollup_bucket(
    cur: psycopg.Cursor[Any], bucket: datetime, report: dict[str, Any]
) -> None:
    params = {"bucket": bucket}
    cur.execute(_ROLLUP_PAYLOAD_SQL, params)
    expected = Counter(tuple(row) for row in cur.fetchall())
    cur.execute(_STORED_ROLLUP_SQL, params)
    actual = Counter(tuple(row) for row in cur.fetchall())
    for item in sorted(set(expected) | set(actual), key=repr):
        if expected[item] == actual[item]:
            continue
        _append_mismatch(
            report,
            "rollup_mismatches",
            {
                "bucket_at": bucket.isoformat(),
                "payload": list(item),
                "expected_count": expected[item],
                "actual_count": actual[item],
            },
            count=abs(expected[item] - actual[item]),
        )


def _verify_population(
    cur: psycopg.Cursor[Any], plans: list[Plan], run_ids: list[int], report: dict[str, Any]
) -> None:
    expected = Counter(plan.population_key for plan in plans)
    cur.execute(
        """SELECT run_id,sample_id,provider,model,voice,dataset_id
           FROM benchmarks_v2.benchmark_observations
           WHERE run_id=ANY(%s) AND benchmark='S2S'""",
        (run_ids,),
    )
    actual = Counter(tuple(row) for row in cur.fetchall())
    _record_counter_diff(report, "observation_population_mismatches", expected, actual)


_PUBLIC_PARITY_PAGE_SQL = """
WITH legacy AS (
  SELECT jsonb_build_array(r.run_id,r.audio_filename,r.provider,r.model,r.voice,n.dataset_id,
                           r.metric_type,r.metric_units,r.metric_value) AS identity,
         count(*)::int AS row_count
  FROM benchmarks_v2.results r
  JOIN benchmarks_v2.runs n ON n.id=r.run_id
  WHERE r.run_id=ANY(%(run_ids)s) AND r.id BETWEEN %(low)s AND %(high)s
    AND r.benchmark='S2S'
    AND r.status='success' AND r.metric_value IS NOT NULL
    AND n.status IN ('succeeded','partial')
  GROUP BY r.run_id,r.audio_filename,r.provider,r.model,r.voice,n.dataset_id,
           r.metric_type,r.metric_units,r.metric_value
), normalized AS (
  SELECT jsonb_build_array(o.run_id,o.sample_id,o.provider,o.model,o.voice,o.dataset_id,
                           e.metric_type,v.unit,v.value) AS identity,
         count(*)::int AS row_count
  FROM benchmarks_v2.benchmark_observations o
  JOIN benchmarks_v2.runs n ON n.id=o.run_id
  JOIN benchmarks_v2.metric_evaluations e ON e.observation_id=o.id
  JOIN benchmarks_v2.metric_values v ON v.metric_evaluation_id=e.id
  WHERE o.run_id=ANY(%(run_ids)s) AND o.benchmark='S2S'
    AND o.status='succeeded' AND e.status='succeeded'
    AND e.metric_version='v1' AND e.evaluation_variant='default'
    AND v.value_key='primary' AND v.value_role='primary'
    AND n.status IN ('succeeded','partial')
  GROUP BY o.run_id,o.sample_id,o.provider,o.model,o.voice,o.dataset_id,
           e.metric_type,v.unit,v.value
)
SELECT COALESCE(l.identity,n.identity),COALESCE(l.row_count,0),COALESCE(n.row_count,0)
FROM legacy l FULL OUTER JOIN normalized n ON l.identity=n.identity
WHERE COALESCE(l.row_count,0) <> COALESCE(n.row_count,0)
ORDER BY 1
"""


def _verify_public_parity_page(
    cur: psycopg.Cursor[Any],
    run_ids: list[int],
    low: int,
    high: int,
    report: dict[str, Any],
) -> None:
    cur.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (_PUBLIC_PARITY_STATEMENT_TIMEOUT,),
    )
    cur.execute(
        _PUBLIC_PARITY_PAGE_SQL,
        {"run_ids": run_ids, "low": low, "high": high},
    )
    for row in cur.fetchall():
        identity = row[0]
        legacy_count = int(row[1])
        normalized_count = int(row[2])
        _append_mismatch(
            report,
            "public_parity_mismatches",
            {
                "identity": identity,
                "legacy_count": legacy_count,
                "normalized_count": normalized_count,
            },
            count=abs(legacy_count - normalized_count),
        )


def _finalize_report(report: dict[str, Any], *, apply: bool) -> None:
    source_skips = _counter_total(report["skipped_by_reason"])
    verification_skips = _counter_total(report["verification_skipped_by_reason"])
    source_conflicts = _counter_total(report["conflicts_by_reason"])
    verification_conflicts = _counter_total(report["verification_conflicts_by_reason"])
    mismatches = sum(int(report[key]) for key in _MISMATCH_KEYS.values())
    report["skip_count"] = source_skips + verification_skips
    report["conflict_count"] = source_conflicts + verification_conflicts
    report["mismatch_count"] = mismatches
    report["backfill_complete"] = (
        report["skip_count"] == 0
        and report["conflict_count"] == 0
        and mismatches == 0
        and (apply or report["eligible"] == 0)
    )
    for key in (
        "skipped_by_reason",
        "conflicts_by_reason",
        "verification_skipped_by_reason",
        "verification_conflicts_by_reason",
    ):
        report[key] = {reason: count for reason, count in report[key].items() if count}


def backfill(
    conn: psycopg.Connection,
    *,
    min_result_id: int,
    max_result_id: int,
    batch_size: int = 100,
    apply: bool = False,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    if min_result_id < 1 or max_result_id < min_result_id or batch_size < 1:
        raise ValueError("invalid id range or batch size")
    report = _report(min_result_id, max_result_id, batch_size)
    progress = reporter or ProgressReporter(
        "apply" if apply else "dry_run", min_result_id, max_result_id, batch_size
    )
    phase = "operation"
    lock_acquired = False
    progress.phase_started(phase, report)
    try:
        phase = "qualifying_run_count"
        progress.phase_started(phase, report)
        progress.total_runs = _qualifying_run_count(conn, min_result_id, max_result_id)
        progress.phase_completed(phase, report)
        if apply:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(hashtextextended(%s,0))", (_LOCK,))
            conn.commit()
            lock_acquired = True

        phase = "source_reconciliation"
        progress.phase_started(phase, report, total_runs=progress.total_runs)
        after = 0
        while True:
            page_skips: Counter[str] = Counter()
            run_ids, rows, complete = _read_complete_page(
                conn,
                min_result_id,
                max_result_id,
                after,
                batch_size,
                page_skips,
            )
            if not run_ids:
                break
            after = run_ids[-1]
            page_conflicts: Counter[str] = Counter()
            plans = _plans(complete, page_skips, page_conflicts)
            missing_schedule = [plan for plan in plans if plan.first.scheduled_at is None]
            page_skips["scheduled_at_missing"] += len(missing_schedule)
            plans = [plan for plan in plans if plan.first.scheduled_at is not None]
            report["source_rows"] += len(rows)
            report["source_runs"] += len(run_ids)
            report["source_groups"] += len(plans)
            report["source_pages"] += 1
            report["skipped_by_reason"].update(page_skips)
            report["conflicts_by_reason"].update(page_conflicts)
            delta = PageDelta()
            if apply:
                with conn.transaction(), conn.cursor() as cur:
                    for plan in plans:
                        delta.record(_reconcile(cur, plan, apply=True))
                    for bucket in sorted(
                        {plan.first.scheduled_at for plan in plans if plan.first.scheduled_at}
                    ):
                        _refresh(cur, bucket)
                        delta.buckets += 1
                _merge_source_delta(report, delta)
            else:
                with conn.transaction(), conn.cursor() as cur:
                    cur.execute("SET TRANSACTION READ ONLY")
                    for plan in plans:
                        delta.record(_reconcile(cur, plan, apply=False))
                _merge_source_delta(report, delta)
            progress.completed_page(
                run_ids,
                rows,
                len(plans),
                report,
                phase=phase,
                committed=apply,
            )
        progress.phase_completed(phase, report)

        phase = "post_write_verification"
        progress.phase_started(phase, report, total_runs=progress.total_runs)
        after = 0
        while True:
            page_skips = Counter()
            run_ids, rows, complete = _read_complete_page(
                conn,
                min_result_id,
                max_result_id,
                after,
                batch_size,
                page_skips,
            )
            if not run_ids:
                break
            after = run_ids[-1]
            page_conflicts = Counter()
            plans = _plans(complete, page_skips, page_conflicts)
            missing_schedule = [plan for plan in plans if plan.first.scheduled_at is None]
            page_skips["scheduled_at_missing"] += len(missing_schedule)
            plans = [plan for plan in plans if plan.first.scheduled_at is not None]
            report["verification_rows"] += len(rows)
            report["verification_runs"] += len(run_ids)
            report["verification_groups"] += len(plans)
            report["verification_pages"] += 1
            report["verification_skipped_by_reason"].update(page_skips)
            report["verification_conflicts_by_reason"].update(page_conflicts)
            delta = PageDelta()
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                for plan in plans:
                    result = _reconcile(cur, plan, apply=False)
                    delta.record(result)
                    if result.disposition in ("eligible", "conflict"):
                        _append_mismatch(
                            report,
                            "payload_mismatches",
                            {
                                "natural_key": list(plan.natural),
                                "reason": result.reason or "observation_missing",
                            },
                        )
                complete_run_ids = sorted({row.run_id for row in complete})
                if complete_run_ids:
                    _verify_population(cur, plans, complete_run_ids, report)
            _merge_verification_delta(report, delta)
            progress.completed_page(run_ids, rows, len(plans), report, phase=phase)
        progress.phase_completed(phase, report)

        phase = "public_parity"
        progress.phase_started(phase, report, total_runs=progress.total_runs)
        after = 0
        while True:
            parity_skips: Counter[str] = Counter()
            run_ids, rows, complete = _read_complete_page(
                conn,
                min_result_id,
                max_result_id,
                after,
                batch_size,
                parity_skips,
            )
            if not run_ids:
                break
            after = run_ids[-1]
            complete_run_ids = sorted({row.run_id for row in complete})
            if complete_run_ids:
                with conn.transaction(), conn.cursor() as cur:
                    cur.execute("SET TRANSACTION READ ONLY")
                    _verify_public_parity_page(
                        cur,
                        complete_run_ids,
                        min_result_id,
                        max_result_id,
                        report,
                    )
            conversations = len(
                {
                    (row.run_id, row.sample_id, row.provider, row.model, row.voice)
                    for row in complete
                }
            )
            progress.completed_page(run_ids, rows, conversations, report, phase=phase)
        progress.phase_completed(phase, report)

        phase = "rollup_verification"
        progress.phase_started(phase, report)
        for buckets in _scheduled_bucket_pages(conn, min_result_id, max_result_id, batch_size):
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                for bucket in buckets:
                    _verify_rollup_bucket(cur, bucket, report)
                    progress.completed_unit(report, phase=phase)
        progress.phase_completed(phase, report)

        _finalize_report(report, apply=apply)
        progress.phase_completed("operation", report)
        return report
    except BaseException:
        status = "cancelled" if isinstance(sys.exception(), _Cancelled) else "failed"
        progress.emit(phase, status, report)
        progress.emit("operation", status, report)
        raise
    finally:
        if lock_acquired:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (_LOCK,))
            conn.commit()


@click.command(name="backfill-normalized-s2s-storage")
@click.option("--min-result-id", type=click.IntRange(1), default=1, show_default=True)
@click.option("--max-result-id", type=click.IntRange(1))
@click.option("--batch-size", type=click.IntRange(1), default=100, show_default=True)
@click.option("--apply", is_flag=True, help="Write immutable normalized rows.")
def backfill_normalized_s2s_storage_cli(
    min_result_id: int, max_result_id: int | None, batch_size: int, apply: bool
) -> None:
    """Reconcile a frozen inclusive legacy S2S result window."""
    if apply and max_result_id is None:
        raise click.UsageError("--apply requires an explicit --max-result-id")
    url = str(get_settings().database_url)
    if url == "postgresql://unused:unused@127.0.0.1:5432/unused":
        raise click.ClickException(
            "DATABASE_URL is required for this production migration; set a non-local production URL"
        )
    with _sigterm(), psycopg.connect(url) as conn:
        if max_result_id is None:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(
                    "SELECT COALESCE(max(id),0) FROM benchmarks_v2.results WHERE benchmark='S2S'"
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("S2S max result query returned no row")
                max_result_id = int(row[0])
        if max_result_id < min_result_id:
            raise click.UsageError("--max-result-id must be >= --min-result-id")
        report = backfill(
            conn,
            min_result_id=min_result_id,
            max_result_id=max_result_id,
            batch_size=batch_size,
            apply=apply,
        )
    click.echo(json.dumps(report, sort_keys=True, default=str))
    if apply and not report["backfill_complete"]:
        raise click.ClickException("backfill incomplete or conflicting")
