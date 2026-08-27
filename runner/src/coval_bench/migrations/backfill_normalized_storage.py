# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # This module intentionally embeds audited SQL statements.
"""Idempotently reconcile legacy STT/TTS rows into normalized storage.

This is deliberately a one-shot operator tool.  It never changes legacy rows
and it refuses to "adopt" a natural observation key owned by dual-write.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from typing import Any

import click
import psycopg
from google.cloud import storage

from coval_bench.config import get_settings
from coval_bench.datasets.loader import _load_manifest
from coval_bench.datasets.manifest import STTManifestItem, TTSManifestItem
from coval_bench.observation_artifacts import (
    prepare_provider_transcript,
    prepare_timing_events,
    upload_provider_transcript,
    upload_timing_events,
)
from coval_bench.registries import Metric
from coval_bench.registries.metrics import MetricValueRole, validate_metric_values

_OBS_NAMESPACE = uuid.UUID("6edc052c-4d05-5b64-9621-cc6f6343b6cd")
_EVAL_NAMESPACE = uuid.UUID("e7f9ce4b-8f66-572c-bf9e-4f67b044422e")
_ARTIFACT_NAMESPACE = uuid.UUID("64478df1-0d13-5996-a3ca-96c421f772c9")
_LOCK = "normalized_storage_backfill"


@dataclass(frozen=True)
class LegacyRow:
    id: int
    run_id: int
    dataset_id: str
    dataset_sha256: str
    scheduled_at: datetime | None
    provider: str
    model: str
    voice: str | None
    benchmark: str
    metric: str
    value: float | None
    unit: str | None
    filename: str | None
    transcript: str | None
    status: str
    error: str | None
    http_version: str | None
    headers_ms: float | None
    created_at: datetime
    ins: float | None
    dels: float | None
    subs: float | None


@dataclass
class Planned:
    rows: list[LegacyRow]
    sample_id: str
    dataset_id: str
    dataset_sha256: str
    benchmark: str
    source_kind: str
    status: str
    error: str | None
    failure_origin: str | None
    artifacts: list[tuple[str, str]]  # (kind, body), no storage side effect

    @property
    def first(self) -> LegacyRow:
        return self.rows[0]

    @property
    def natural(self) -> tuple[Any, ...]:
        r = self.first
        return (r.run_id, self.sample_id, r.provider, r.model, r.voice)

    @property
    def id(self) -> uuid.UUID:
        return uuid.uuid5(_OBS_NAMESPACE, "|".join(map(str, self.natural)))


def _valid_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _manifest_sha(dataset_id: str) -> str | None:
    try:
        raw = files("coval_bench.datasets.manifests").joinpath(f"{dataset_id}.json").read_bytes()
    except FileNotFoundError:
        return None
    return hashlib.sha256(raw).hexdigest()


def _legacy_sample(dataset_id: str, filename: str) -> str:
    return f"legacy:{uuid.uuid5(_OBS_NAMESPACE, f'stt|{dataset_id}|{filename}')}"


def _stt_sample(dataset_id: str, dataset_sha256: str, filename: str) -> str | None:
    if not filename:
        return None
    try:
        manifest = _load_manifest(dataset_id)
    except FileNotFoundError:
        return _legacy_sample(dataset_id, filename)
    # A filename alone is not provenance.  Packaged identities are valid only
    # when the historical run pins these exact manifest bytes.
    if _manifest_sha(dataset_id) != dataset_sha256:
        return _legacy_sample(dataset_id, filename)
    matches = [
        x
        for x in manifest.items
        if isinstance(x, STTManifestItem) and x.path.rsplit("/", 1)[-1] == filename
    ]
    if len(matches) != 1:
        return _legacy_sample(dataset_id, filename)
    item = matches[0]
    return item.sample_id or item.path


def _tts_sample(dataset_id: str, sha: str, prompt: str) -> str | None:
    # A current tts-v1 prompt is usable only if the historical run proves it
    # used the exact packaged manifest bytes.
    if dataset_id != "tts-v1" or _manifest_sha(dataset_id) != sha:
        return None
    try:
        manifest = _load_manifest(dataset_id)
    except FileNotFoundError:
        return None
    matches = [
        x for x in manifest.items if isinstance(x, TTSManifestItem) and x.transcript == prompt
    ]
    return matches[0].testcase_id if len(matches) == 1 else None


def _report() -> dict[str, Any]:
    return {
        "window": {},
        "source_rows": 0,
        "source_groups": 0,
        "eligible": 0,
        "created": 0,
        "reconciled": 0,
        "live_owned": 0,
        "evaluations": 0,
        "values": 0,
        "artifacts": 0,
        "buckets": 0,
        "skipped_by_reason": Counter(),
        "parity_mismatches": [],
        "rollup_mismatches": [],
        "cutover_ready": False,
    }


_ROWS_SQL = """
SELECT r.id,r.run_id,n.dataset_id,n.dataset_sha256,n.scheduled_at,r.provider,r.model,r.voice,
 r.benchmark,r.metric_type,r.metric_value,r.metric_units,r.audio_filename,r.transcript,r.status,
 r.error,r.http_version,r.submit_to_headers_ms,r.created_at,r.wer_insertions_pct,r.wer_deletions_pct,r.wer_substitutions_pct
FROM benchmarks_v2.results r JOIN benchmarks_v2.runs n ON n.id=r.run_id
WHERE r.id BETWEEN %s AND %s AND r.benchmark IN ('STT','TTS') ORDER BY r.id
"""


def _rows(conn: psycopg.Connection, low: int, high: int) -> list[LegacyRow]:
    with conn.cursor() as cur:
        cur.execute(_ROWS_SQL, (low, high))
        return [LegacyRow(*row) for row in cur.fetchall()]


def _complete_window_rows(
    conn: psycopg.Connection, rows: list[LegacyRow], low: int, high: int, skipped: Counter[str]
) -> list[LegacyRow]:
    """A result window is safe only if each selected run is wholly present and terminal."""
    run_ids = sorted({row.run_id for row in rows})
    if not run_ids:
        return rows
    with conn.cursor() as cur:
        cur.execute(
            """SELECT n.id, min(r.id), max(r.id), n.status FROM benchmarks_v2.runs n
               JOIN benchmarks_v2.results r ON r.run_id=n.id WHERE n.id=ANY(%s)
               GROUP BY n.id,n.status""",
            (run_ids,),
        )
        bad: set[int] = set()
        for run_id, first_id, last_id, status in cur.fetchall():
            if first_id < low or last_id > high:
                skipped["split_window_run"] += 1
                bad.add(run_id)
            elif status not in ("succeeded", "partial", "failed"):
                skipped["run_not_terminal"] += 1
                bad.add(run_id)
    return [row for row in rows if row.run_id not in bad]


def _stt_plans(rows: Iterable[LegacyRow], skipped: Counter[str]) -> list[Planned]:
    grouped: dict[tuple[Any, ...], list[LegacyRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.run_id, r.provider, r.model, r.voice, r.filename)].append(r)
    plans = []
    for key, group in grouped.items():
        filename = key[-1]
        if not filename:
            skipped["stt_sample_unrecoverable"] += 1
            continue
        r = group[0]
        if not _valid_sha(r.dataset_sha256):
            skipped["invalid_dataset_sha"] += 1
            continue
        sample = _stt_sample(r.dataset_id, r.dataset_sha256, filename)
        if not sample:
            skipped["stt_sample_ambiguous"] += 1
            continue
        transcripts = {x.transcript for x in group if x.transcript is not None}
        if len(transcripts) > 1:
            skipped["stt_provider_transcript_ambiguous"] += 1
            continue
        artifacts = [] if len(transcripts) != 1 else [("provider_transcript", transcripts.pop())]
        statuses = {x.status for x in group}
        failed = statuses == {"failed"}
        errors = {x.error.strip() for x in group if x.error and x.error.strip()}
        if failed and len(errors) != 1:
            skipped["failed_observation_error_unrecoverable"] += 1
            continue
        if not _metrics_are_valid(group, skipped):
            continue
        # Legacy rows have no failure-origin column.  A provider error was
        # copied onto every STT metric, so multiple distinct failed metrics
        # carrying one identical error are the only defensible proof.  A
        # single failed metric is ambiguous and remains a failed evaluation
        # under a succeeded observation.
        failed_metrics = {x.metric for x in group if x.status == "failed"}
        if failed and len(failed_metrics) < 2:
            skipped["observation_failure_origin_unrecoverable"] += 1
            continue
        error = next(iter(errors)) if failed else None
        plans.append(
            Planned(
                group,
                sample,
                r.dataset_id,
                r.dataset_sha256,
                "STT",
                "dataset_audio",
                "failed" if error else "succeeded",
                error,
                "provider" if error else None,
                artifacts,
            )
        )
    return plans


def _tts_plans(rows: Iterable[LegacyRow], skipped: Counter[str]) -> list[Planned]:
    # TTFA records are anchors; its transcript is the historical prompt.
    source_rows = list(rows)
    anchors = [r for r in source_rows if r.metric == str(Metric.TTFA)]
    plans = []
    for a in anchors:
        if not a.transcript:
            skipped["tts_anchor_prompt_missing"] += 1
            continue
        if not a.filename:
            skipped["tts_anchor_filename_missing"] += 1
            continue
        if not _valid_sha(a.dataset_sha256):
            skipped["invalid_dataset_sha"] += 1
            continue
        sample = _tts_sample(a.dataset_id, a.dataset_sha256, a.transcript)
        if not sample:
            skipped["tts_dataset_sha_unrecoverable"] += 1
            continue
        base = (a.run_id, a.provider, a.model, a.voice)
        attached = [
            r
            for r in source_rows
            if (r.run_id, r.provider, r.model, r.voice) == base
            and (
                (
                    r.metric in (str(Metric.TTFA_ROUNDTRIP), str(Metric.TTFA_LEADING_SILENCE))
                    and r.transcript == a.transcript
                )
                or (r.metric == str(Metric.WER) and a.filename and r.filename == a.filename)
                or r.id == a.id
            )
        ]
        # A duplicate TTFA prompt means there is no safe identity for attachment.
        if (
            sum(
                r.metric == str(Metric.TTFA) and r.transcript == a.transcript
                for r in source_rows
                if (r.run_id, r.provider, r.model, r.voice) == base
            )
            != 1
        ):
            skipped["tts_anchor_ambiguous"] += 1
            continue
        component_counts = Counter(r.metric for r in attached)
        if any(
            component_counts[metric] > 1
            for metric in ("TTFARoundtrip", "TTFALeadingSilence", "WER")
        ):
            skipped["tts_component_or_wer_ambiguous"] += 1
            continue
        if (
            a.filename
            and sum(
                r.metric == "TTFA"
                and r.filename == a.filename
                and (r.run_id, r.provider, r.model, r.voice) == base
                for r in anchors
            )
            != 1
        ):
            skipped["tts_filename_cross_anchor_ambiguous"] += 1
            continue
        failed = all(r.status == "failed" for r in attached)
        errors = {r.error.strip() for r in attached if r.error and r.error.strip()}
        if failed and len(errors) != 1:
            skipped["failed_observation_error_unrecoverable"] += 1
            continue
        if failed and not (
            len(attached) == 1
            and attached[0].metric == str(Metric.TTFA)
            and errors == {f"no {Metric.TTFA} produced"}
        ):
            skipped["observation_failure_origin_unrecoverable"] += 1
            continue
        if not _metrics_are_valid(attached, skipped):
            continue
        # A provider error and a metric-level TTFA failure are indistinguishable
        # in the historical single-row TTS shape.  Preserve the metric failure,
        # but do not fabricate observation failure provenance.
        error = None
        artifacts = (
            []
            if a.value is None
            else [
                (
                    "timing_events",
                    json.dumps({"ttfa_ms": a.value}, sort_keys=True, separators=(",", ":")),
                )
            ]
        )
        plans.append(
            Planned(
                attached,
                sample,
                a.dataset_id,
                a.dataset_sha256,
                "TTS",
                "generated_audio",
                "failed" if error else "succeeded",
                error,
                "provider" if error else None,
                artifacts,
            )
        )
    claimed_ids = {row.id for plan in plans for row in plan.rows}
    unclaimed_ids = {row.id for row in source_rows} - claimed_ids
    if unclaimed_ids:
        skipped["tts_source_rows_unclaimed"] += len(unclaimed_ids)
    return plans


def _values(rows: list[LegacyRow]) -> list[tuple[str, str, float, str, str]]:
    out = []
    by_metric: dict[str, list[LegacyRow]] = defaultdict(list)
    for r in rows:
        by_metric[
            "TTFA" if r.metric in ("TTFARoundtrip", "TTFALeadingSilence") else r.metric
        ].append(r)
    for metric, group in by_metric.items():
        # TTFA components are legacy rows in the TTFA evaluation, but only the
        # actual TTFA row is its primary value.  Never let source row order pick
        # a component as primary.
        primary = next(
            (
                r
                for r in group
                if r.metric == metric and r.status == "success" and r.value is not None
            ),
            None,
        )
        if primary is None:
            continue
        primary_value = primary.value
        if primary_value is None:  # narrowed above; retain a runtime guard for malformed rows.
            continue
        out.append((metric, "primary", float(primary_value), primary.unit or "", "primary"))
        if metric == "WER":
            out.extend(
                (metric, key, float(v), "percent", "component")
                for key, v in (
                    ("insertions", primary.ins),
                    ("deletions", primary.dels),
                    ("substitutions", primary.subs),
                )
                if v is not None
            )
        if metric == "TTFA":
            components = {r.metric: r for r in group}
            roundtrip = components.get("TTFARoundtrip")
            silence = components.get("TTFALeadingSilence")
            if (
                roundtrip is not None
                and silence is not None
                and roundtrip.status == "success"
                and silence.status == "success"
                and roundtrip.value is not None
                and silence.value is not None
            ):
                out.extend(
                    [
                        (
                            metric,
                            "roundtrip",
                            float(roundtrip.value),
                            "milliseconds",
                            "component",
                        ),
                        (
                            metric,
                            "leading_silence",
                            float(silence.value),
                            "milliseconds",
                            "component",
                        ),
                    ]
                )
    return out


def _metric_groups(rows: list[LegacyRow]) -> dict[str, list[LegacyRow]]:
    grouped: dict[str, list[LegacyRow]] = defaultdict(list)
    for row in rows:
        grouped[
            "TTFA" if row.metric in ("TTFARoundtrip", "TTFALeadingSilence") else row.metric
        ].append(row)
    return grouped


def _metrics_are_valid(rows: list[LegacyRow], skipped: Counter[str]) -> bool:
    """Reject ambiguous legacy metric payloads before they become terminal rows."""
    try:
        values = _values(rows)
        for metric, group in _metric_groups(rows).items():
            original_counts = Counter(r.metric for r in group)
            if any(count > 1 for count in original_counts.values()):
                raise ValueError("duplicate legacy metric row")
            if metric == str(Metric.TTFA):
                anchors = [r for r in group if r.metric == str(Metric.TTFA)]
                if len(anchors) != 1:
                    raise ValueError("TTFA evaluation requires exactly one anchor")
                anchor = anchors[0]
                components = [
                    r
                    for r in group
                    if r.metric in (str(Metric.TTFA_ROUNDTRIP), str(Metric.TTFA_LEADING_SILENCE))
                ]
                if components and (
                    len(components) != 2
                    or any(r.status != "success" or r.value is None for r in components)
                ):
                    raise ValueError("invalid TTFA component rows")
                if components and (anchor.status != "success" or anchor.value is None):
                    raise ValueError("TTFA components require a successful valued anchor")
            primary_rows = [
                r
                for r in group
                if r.metric == metric and r.status == "success" and r.value is not None
            ]
            if len(primary_rows) > 1:
                raise ValueError("duplicate primary")
            metric_values = [item for item in values if item[0] == metric]
            if metric_values:
                if any(r.status == "failed" for r in group):
                    raise ValueError("successful metric payload contains failed source rows")
                validate_metric_values(
                    metric,
                    "v1",
                    tuple(
                        (key, unit, value, MetricValueRole(role))
                        for _, key, value, unit, role in metric_values
                    ),
                )
            elif any(r.status == "success" for r in group) and any(
                r.status == "failed" for r in group
            ):
                raise ValueError("mixed metric outcome has no primary value")
            elif any(r.status == "failed" for r in group):
                errors = {r.error.strip() for r in group if r.error and r.error.strip()}
                if len(errors) != 1:
                    raise ValueError("failed metric has no provable error")
    except (TypeError, ValueError):
        skipped["metric_payload_invalid"] += 1
        return False
    return True


def _stored_plan_matches(
    cur: psycopg.Cursor[tuple[Any, ...]], plan: Planned, observation_id: uuid.UUID
) -> bool:
    """Compare the legacy-derived immutable payload, never merely its parent ID."""
    row = plan.first
    cur.execute(
        """SELECT run_id,dataset_id,dataset_sha256,sample_id,provider,model,voice,benchmark,
                  source_kind,transport_protocol,submit_to_headers_ms,captured_at,status,error,failure_origin
           FROM benchmarks_v2.benchmark_observations WHERE id=%s""",
        (observation_id,),
    )
    actual = cur.fetchone()
    expected = (
        row.run_id,
        plan.dataset_id,
        plan.dataset_sha256,
        plan.sample_id,
        row.provider,
        row.model,
        row.voice,
        plan.benchmark,
        plan.source_kind,
        row.http_version,
        row.headers_ms,
        row.created_at,
        plan.status,
        plan.error,
        plan.failure_origin,
    )
    if actual != expected:
        return False
    cur.execute(
        """SELECT e.metric_type,e.metric_version,e.evaluation_variant,e.executor,e.status,
                  e.started_at,e.finished_at,e.error,v.value_key,v.unit,v.value,v.value_role
           FROM benchmarks_v2.metric_evaluations e LEFT JOIN benchmarks_v2.metric_values v
             ON v.metric_evaluation_id=e.id WHERE e.observation_id=%s
           ORDER BY e.metric_type,v.value_key""",
        (observation_id,),
    )
    actual_values = [tuple(item) for item in cur.fetchall()]
    expected_values: list[tuple[Any, ...]] = []
    values_by_metric: dict[str, list[tuple[str, str, float, str, str]]] = defaultdict(list)
    for datum in _values(plan.rows):
        values_by_metric[datum[0]].append(datum)
    for metric, values in values_by_metric.items():
        for _, key, metric_value, unit, role in values:
            expected_values.append(
                (
                    metric,
                    "v1",
                    "default",
                    "inline",
                    "succeeded",
                    row.created_at,
                    row.created_at,
                    None,
                    key,
                    unit,
                    metric_value,
                    role,
                )
            )
    for metric, metric_rows in _metric_groups(plan.rows).items():
        if (
            metric not in values_by_metric
            and metric_rows
            and all(r.status == "failed" for r in metric_rows)
        ):
            error = next(r.error for r in metric_rows if r.error)
            expected_values.append(
                (
                    metric,
                    "v1",
                    "default",
                    "inline",
                    "failed",
                    row.created_at,
                    row.created_at,
                    error,
                    None,
                    None,
                    None,
                    None,
                )
            )
    if sorted(actual_values, key=repr) != sorted(expected_values, key=repr):
        return False

    expected_artifacts: list[tuple[Any, ...]] = []
    for kind, body in plan.artifacts:
        artifact_type, payload, extension, _, schema_name = (
            prepare_provider_transcript(body)
            if kind == "provider_transcript"
            else prepare_timing_events(json.loads(body))
        )
        digest = hashlib.sha256(payload).hexdigest()
        key = f"observation-artifacts/v1/{artifact_type}/{digest[:2]}/{digest}.{extension}"
        expected_artifacts.append(
            (
                str(artifact_type),
                schema_name,
                "v1",
                key,
                digest,
                len(payload),
                None,
            )
        )
    cur.execute(
        """SELECT artifact_type,schema_name,schema_version,
                  regexp_replace(gcs_uri, '^gs://[^/]+/', ''),
                  content_sha256,size_bytes,duration_ms
           FROM benchmarks_v2.observation_artifacts WHERE observation_id=%s
           ORDER BY artifact_type""",
        (observation_id,),
    )
    actual_artifacts = [tuple(item) for item in cur.fetchall()]
    if actual_artifacts != sorted(expected_artifacts, key=repr):
        return False

    expected_inputs: list[tuple[Any, ...]] = []
    artifact_by_kind = {
        kind: expected
        for (kind, _), expected in zip(plan.artifacts, expected_artifacts, strict=True)
    }
    evaluated_metrics = {metric for metric, *_ in _values(plan.rows)}
    evaluated_metrics.update(
        metric
        for metric, metric_rows in _metric_groups(plan.rows).items()
        if metric_rows and all(row.status == "failed" for row in metric_rows)
    )
    for metric in evaluated_metrics:
        input_kind = (
            "provider_transcript"
            if plan.benchmark == "STT" and metric == "WER"
            else "timing_events"
            if plan.benchmark == "TTS" and metric == "TTFA"
            else None
        )
        artifact = artifact_by_kind.get(input_kind) if input_kind is not None else None
        if artifact is not None:
            expected_inputs.append(
                (
                    metric,
                    "raw" if metric == "WER" else "timing",
                    0,
                    artifact[0],
                    artifact[4],
                )
            )
    cur.execute(
        """SELECT e.metric_type,i.input_role,i.input_order,a.artifact_type,a.content_sha256
           FROM benchmarks_v2.metric_evaluation_inputs i
           JOIN benchmarks_v2.metric_evaluations e ON e.id=i.metric_evaluation_id
           LEFT JOIN benchmarks_v2.observation_artifacts a
             ON a.id=i.observation_artifact_id
           WHERE e.observation_id=%s
           ORDER BY e.metric_type,i.input_role,i.input_order""",
        (observation_id,),
    )
    return [tuple(item) for item in cur.fetchall()] == sorted(expected_inputs, key=repr)


def _insert_plan(
    conn: psycopg.Connection,
    plan: Planned,
    apply: bool,
    bucket: str | None,
    client: storage.Client | None,
    report: dict[str, Any],
) -> None:
    r = plan.first
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id FROM benchmarks_v2.benchmark_observations
               WHERE run_id=%s AND sample_id=%s AND provider=%s AND model=%s
                 AND voice IS NOT DISTINCT FROM %s""",
            plan.natural,
        )
        found = cur.fetchone()
        if found and found[0] != plan.id:
            report["live_owned"] += 1
            if _stored_plan_matches(cur, plan, found[0]):
                report["reconciled"] += 1
            else:
                report["parity_mismatches"].append(
                    {"natural_key": list(plan.natural), "reason": "live_owned_payload_mismatch"}
                )
            return
        if found:
            if _stored_plan_matches(cur, plan, found[0]):
                report["reconciled"] += 1
            else:
                report["parity_mismatches"].append(
                    {"natural_key": list(plan.natural), "reason": "backfill_payload_mismatch"}
                )
            return
        report["eligible"] += 1
        if not apply:
            return
        cur.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
        (id,run_id,dataset_id,dataset_sha256,sample_id,provider,model,voice,benchmark,source_kind,transport_protocol,submit_to_headers_ms,captured_at,status,error,failure_origin)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                plan.id,
                r.run_id,
                plan.dataset_id,
                plan.dataset_sha256,
                plan.sample_id,
                r.provider,
                r.model,
                r.voice,
                plan.benchmark,
                plan.source_kind,
                r.http_version,
                r.headers_ms,
                r.created_at,
                plan.status,
                plan.error,
                plan.failure_origin,
            ),
        )
        report["created"] += 1
        artifact_ids = {}
        for kind, body in plan.artifacts:
            payload = (
                prepare_provider_transcript(body)[1]
                if kind == "provider_transcript"
                else prepare_timing_events(json.loads(body))[1]
            )
            aid = uuid.uuid5(
                _ARTIFACT_NAMESPACE, f"{plan.id}|{kind}|{hashlib.sha256(payload).hexdigest()}"
            )
            if bucket is None or client is None:
                raise click.ClickException(
                    "BENCHMARK_ARTIFACT_BUCKET is required for recoverable artifacts"
                )
            art = (
                upload_provider_transcript(client, bucket, body)
                if kind == "provider_transcript"
                else upload_timing_events(client, bucket, json.loads(body))
            )
            cur.execute(
                """INSERT INTO benchmarks_v2.observation_artifacts
                (id,observation_id,artifact_type,schema_name,schema_version,gcs_uri,content_sha256,size_bytes,duration_ms)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    aid,
                    plan.id,
                    art.artifact_type,
                    art.schema_name,
                    art.schema_version,
                    art.gcs_uri,
                    art.content_sha256,
                    art.size_bytes,
                    art.duration_ms,
                ),
            )
            artifact_ids[kind] = aid
            report["artifacts"] += 1
        grouped: dict[str, list[tuple[str, str, float, str, str]]] = defaultdict(list)
        for row in _values(plan.rows):
            grouped[row[0]].append(row)
        for metric, values in grouped.items():
            eid = uuid.uuid5(_EVAL_NAMESPACE, f"{plan.id}|{metric}|v1|default")
            cur.execute(
                "INSERT INTO benchmarks_v2.metric_evaluations (id,observation_id,metric_type,metric_version,evaluation_variant,executor,status) VALUES (%s,%s,%s,'v1','default','inline','queued')",
                (eid, plan.id, metric),
            )
            input_id = artifact_ids.get(
                "provider_transcript"
                if plan.benchmark == "STT" and metric == "WER"
                else "timing_events"
                if plan.benchmark == "TTS" and metric == "TTFA"
                else ""
            )
            if input_id is not None:
                cur.execute(
                    "INSERT INTO benchmarks_v2.metric_evaluation_inputs (metric_evaluation_id,observation_artifact_id,input_role,input_order) VALUES (%s,%s,%s,0)",
                    (eid, input_id, "raw" if metric == "WER" else "timing"),
                )
            cur.execute(
                "UPDATE benchmarks_v2.metric_evaluations SET status='running',started_at=%s WHERE id=%s",
                (r.created_at, eid),
            )
            cur.executemany(
                "INSERT INTO benchmarks_v2.metric_values (metric_evaluation_id,value_key,unit,value,value_role) VALUES (%s,%s,%s,%s,%s)",
                [(eid, key, unit, value, role) for _, key, value, unit, role in values],
            )
            cur.execute(
                "UPDATE benchmarks_v2.metric_evaluations SET status='succeeded',finished_at=%s WHERE id=%s",
                (r.created_at, eid),
            )
            report["values"] += len(values)
            report["evaluations"] += 1
        for metric, rows in _metric_groups(plan.rows).items():
            if metric in grouped or not rows or not all(row.status == "failed" for row in rows):
                continue
            error = next(row.error for row in rows if row.error)
            eid = uuid.uuid5(_EVAL_NAMESPACE, f"{plan.id}|{metric}|v1|default")
            cur.execute(
                "INSERT INTO benchmarks_v2.metric_evaluations (id,observation_id,metric_type,metric_version,evaluation_variant,executor,status) VALUES (%s,%s,%s,'v1','default','inline','queued')",
                (eid, plan.id, metric),
            )
            input_id = artifact_ids.get(
                "provider_transcript"
                if plan.benchmark == "STT" and metric == "WER"
                else "timing_events"
                if plan.benchmark == "TTS" and metric == "TTFA"
                else ""
            )
            if input_id is not None:
                cur.execute(
                    "INSERT INTO benchmarks_v2.metric_evaluation_inputs (metric_evaluation_id,observation_artifact_id,input_role,input_order) VALUES (%s,%s,%s,0)",
                    (eid, input_id, "raw" if metric == "WER" else "timing"),
                )
            cur.execute(
                "UPDATE benchmarks_v2.metric_evaluations SET status='failed',started_at=%s,finished_at=%s,error=%s WHERE id=%s",
                (r.created_at, r.created_at, error, eid),
            )
            report["evaluations"] += 1


def _refresh_bucket(cur: psycopg.Cursor[tuple[Any, ...]], bucket_at: datetime) -> None:
    """Use the RunWriter delete/insert contract over every normalized row."""
    params = {"bucket": bucket_at}
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended('metric_values_by_bucket', extract(epoch FROM %(bucket)s::timestamptz)::bigint))",
        params,
    )
    cur.execute(
        "DELETE FROM benchmarks_v2.metric_values_by_bucket WHERE bucket_at=%(bucket)s",
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
WHERE e.status='succeeded' AND r.status IN ('succeeded','partial')
  AND r.scheduled_at=%(bucket)s
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
WHERE bucket_at=%(bucket)s
"""


def _rollup_mismatches(
    conn: psycopg.Connection, buckets: Iterable[datetime]
) -> list[dict[str, Any]]:
    """Compare materialized rollups with a fresh aggregate over immutable values."""
    mismatches: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        for bucket_at in sorted(set(buckets)):
            params = {"bucket": bucket_at}
            cur.execute(_ROLLUP_PAYLOAD_SQL, params)
            expected = sorted((tuple(row) for row in cur.fetchall()), key=repr)
            cur.execute(_STORED_ROLLUP_SQL, params)
            actual = sorted((tuple(row) for row in cur.fetchall()), key=repr)
            if actual != expected:
                mismatches.append(
                    {
                        "bucket_at": bucket_at.isoformat(),
                        "expected_rows": len(expected),
                        "actual_rows": len(actual),
                        "reason": "stored_bucket_payload_mismatch",
                    }
                )
    return mismatches


def backfill(
    conn: psycopg.Connection,
    *,
    min_result_id: int,
    max_result_id: int,
    batch_size: int,
    apply: bool,
    artifact_bucket: str | None = None,
) -> dict[str, Any]:
    if min_result_id <= 0 or max_result_id < min_result_id or batch_size <= 0:
        raise ValueError("invalid id range or batch size")
    report = _report()
    report["window"] = {
        "min_result_id": min_result_id,
        "max_result_id": max_result_id,
        "batch_size": batch_size,
    }
    rows = _rows(conn, min_result_id, max_result_id)
    report["source_rows"] = len(rows)
    rows = _complete_window_rows(
        conn, rows, min_result_id, max_result_id, report["skipped_by_reason"]
    )
    plans = _stt_plans(
        (r for r in rows if r.benchmark == "STT"), report["skipped_by_reason"]
    ) + _tts_plans((r for r in rows if r.benchmark == "TTS"), report["skipped_by_reason"])
    report["source_groups"] = len(plans)
    missing_schedule = sum(plan.first.scheduled_at is None for plan in plans)
    if missing_schedule:
        report["skipped_by_reason"]["scheduled_at_missing"] += missing_schedule
        plans = [plan for plan in plans if plan.first.scheduled_at is not None]
    buckets = {plan.first.scheduled_at for plan in plans}
    scheduled_buckets = {bucket for bucket in buckets if bucket is not None}
    if not apply:
        for p in plans:
            _insert_plan(conn, p, False, artifact_bucket, None, report)
        report["rollup_mismatches"] = _rollup_mismatches(conn, scheduled_buckets)
        report["skipped_by_reason"] = dict(report["skipped_by_reason"])
        report["cutover_ready"] = (
            not report["eligible"]
            and not report["parity_mismatches"]
            and not report["rollup_mismatches"]
            and not report["skipped_by_reason"]
        )
        return report
    # End planning's implicit transaction before taking a *session* lock.  Each
    # transaction below is now a real commit boundary, not a savepoint.
    conn.commit()
    needs_artifacts = any(p.artifacts for p in plans)
    if needs_artifacts and not artifact_bucket:
        raise click.ClickException(
            "BENCHMARK_ARTIFACT_BUCKET is required for recoverable artifacts"
        )
    client = storage.Client() if needs_artifacts else None
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(hashtextextended(%s,0))", (_LOCK,))
    # ``pg_advisory_lock`` is session scoped.  Commit the implicit SELECT
    # transaction now so each following ``conn.transaction`` is top-level.
    conn.commit()
    try:
        for start in range(0, len(plans), batch_size):
            with conn.transaction():
                for p in plans[start : start + batch_size]:
                    _insert_plan(conn, p, True, artifact_bucket, client, report)
                buckets = {p.first.scheduled_at for p in plans[start : start + batch_size]}
                for bucket_at in buckets:
                    if bucket_at is None:  # guarded by the pre-write plan filter above.
                        raise RuntimeError("scheduled_at changed during immutable plan execution")
                    with conn.cursor() as cur:
                        _refresh_bucket(cur, bucket_at)
                    report["buckets"] += 1
        # The decision is based on a fresh immutable snapshot, not optimistic
        # INSERT success.  This also detects trigger/DB representation drift.
        with conn.cursor() as cur:
            for plan in plans:
                cur.execute(
                    """SELECT id FROM benchmarks_v2.benchmark_observations
                       WHERE run_id=%s AND sample_id=%s AND provider=%s AND model=%s
                         AND voice IS NOT DISTINCT FROM %s""",
                    plan.natural,
                )
                found = cur.fetchone()
                reason = (
                    "post_write_observation_missing"
                    if found is None
                    else "post_write_payload_mismatch"
                    if not _stored_plan_matches(cur, plan, found[0])
                    else None
                )
                if reason is not None:
                    report["parity_mismatches"].append(
                        {"natural_key": list(plan.natural), "reason": reason}
                    )
        report["rollup_mismatches"] = _rollup_mismatches(conn, scheduled_buckets)
        report["skipped_by_reason"] = dict(report["skipped_by_reason"])
        report["cutover_ready"] = (
            not report["parity_mismatches"]
            and not report["rollup_mismatches"]
            and not report["skipped_by_reason"]
        )
        return report
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (_LOCK,))
        conn.commit()


@click.command(name="backfill-normalized-storage")
@click.option("--min-result-id", type=click.IntRange(1), default=1, show_default=True)
@click.option("--max-result-id", type=click.IntRange(1))
@click.option("--batch-size", type=click.IntRange(1), default=100, show_default=True)
@click.option("--apply", is_flag=True, help="Write immutable normalized rows.")
def backfill_normalized_storage_cli(
    min_result_id: int, max_result_id: int | None, batch_size: int, apply: bool
) -> None:
    """Reconcile a frozen inclusive legacy-result window into normalized tables."""
    if apply and max_result_id is None:
        raise click.UsageError("--apply requires an explicit --max-result-id")
    url = str(get_settings().database_url)
    with psycopg.connect(url) as conn:
        if max_result_id is None:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(max(id),0) FROM benchmarks_v2.results")
                row = cur.fetchone()
                if row is None:  # pragma: no cover - aggregate SELECT always returns one row.
                    raise RuntimeError("max result id query returned no row")
                max_result_id = row[0]
        if max_result_id < min_result_id:
            raise click.UsageError("--max-result-id must be >= --min-result-id")
        report = backfill(
            conn,
            min_result_id=min_result_id,
            max_result_id=max_result_id,
            batch_size=batch_size,
            apply=apply,
            artifact_bucket=os.getenv("BENCHMARK_ARTIFACT_BUCKET"),
        )
    click.echo(json.dumps(report, sort_keys=True, default=str))
    if apply and not report["cutover_ready"]:
        raise click.ClickException("backfill incomplete or conflicting")
