# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Safety and real-Postgres coverage for the normalized storage backfill."""

from __future__ import annotations

import hashlib
import io
import json
import signal
from collections import Counter
from contextlib import nullcontext
from dataclasses import astuple, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import click
import psycopg
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from click.testing import CliRunner
from google.cloud import storage as gcs_storage
from pytest_postgresql.factories import postgresql

from coval_bench.db.models import ObservationArtifact, ObservationArtifactType
from coval_bench.migrations import backfill_normalized_storage as migration
from coval_bench.migrations.backfill_normalized_storage import (
    LegacyRow,
    _metrics_are_valid,
    _tts_plans,
    _values,
    backfill,
    backfill_normalized_storage_cli,
)
from coval_bench.observation_artifacts import (
    prepare_provider_transcript,
    prepare_timing_events,
)

backfill_pg = postgresql("pg_proc")
_INI_PATH = Path(__file__).parents[2] / "alembic.ini"
_NOW = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
_SHA = "a" * 64
_WINDOW_START = _NOW - timedelta(hours=167)
_WINDOW_END = _NOW + timedelta(hours=1)


def _dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    auth = f"{info.user}:{info.password}@" if info.password else f"{info.user}@"
    return (
        f"postgresql://{auth}{info.host or 'localhost'}:{info.port or 5432}/{info.dbname or 'test'}"
    )


def _migrate(conn: psycopg.Connection[Any]) -> None:
    config = AlembicConfig(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", _dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    alembic_command.upgrade(config, "head")


def _insert_run(
    conn: psycopg.Connection[Any],
    *,
    dataset_id: str = "historical-stt",
    dataset_sha256: str = _SHA,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.runs
               (started_at,finished_at,runner_sha,dataset_id,dataset_sha256,status,scheduled_at)
               VALUES (%s,%s,'runner-sha',%s,%s,'succeeded',%s)
               RETURNING id""",
            (_NOW, _NOW, dataset_id, dataset_sha256, _NOW),
        )
        row = cur.fetchone()
        assert row is not None
    conn.commit()
    return int(row[0])


def _insert_wer(
    conn: psycopg.Connection[Any],
    run_id: int,
    filename: str,
    *,
    transcript: str = "words",
    created_at: datetime = _NOW,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.results
               (run_id,provider,model,benchmark,metric_type,metric_value,metric_units,
                audio_filename,transcript,status,error,created_at,wer_insertions_pct,
                wer_deletions_pct,wer_substitutions_pct)
               VALUES (%s,'provider','model','STT','WER',6,'percent',%s,%s,'success',
                       NULL,%s,1,2,3) RETURNING id""",
            (run_id, filename, transcript, created_at),
        )
        row = cur.fetchone()
        assert row is not None
    conn.commit()
    return int(row[0])


class _Blob:
    def __init__(self, key: str) -> None:
        self.key = key
        self.metadata: dict[str, str] | None = None
        self.payload = b""

    def upload_from_string(
        self, payload: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        assert content_type == "application/json"
        assert if_generation_match == 0
        self.payload = payload


class _Bucket:
    def __init__(self) -> None:
        self.blobs: dict[str, _Blob] = {}

    def blob(self, key: str) -> _Blob:
        return self.blobs.setdefault(key, _Blob(key))

    def test_iam_permissions(self, permissions: list[str]) -> list[str]:
        return permissions


class _Storage:
    def __init__(self) -> None:
        self.value = _Bucket()

    def bucket(self, name: str) -> _Bucket:
        assert name == "backfill-artifacts"
        return self.value


def _row(
    metric: str,
    value: float | None,
    *,
    status: str = "success",
    filename: str | None = "a.wav",
    transcript: str = "prompt",
) -> LegacyRow:
    return LegacyRow(
        1,
        1,
        "tts-v1",
        _SHA,
        _NOW,
        "provider",
        "model",
        "voice",
        "TTS",
        metric,
        value,
        "milliseconds" if metric.startswith("TTFA") else "percent",
        filename,
        transcript,
        status,
        "failed" if status == "failed" else None,
        None,
        None,
        _NOW,
        1.0,
        2.0,
        3.0,
    )


def test_ttfa_components_and_wer_components_follow_normalized_contract() -> None:
    values = _values(
        [
            _row("TTFARoundtrip", 10.0),
            _row("TTFA", 12.0),
            _row("TTFALeadingSilence", 2.0),
            _row("WER", 6.0),
        ]
    )
    assert ("TTFA", "primary", 12.0, "milliseconds", "primary") in values
    assert ("TTFA", "roundtrip", 10.0, "milliseconds", "component") in values
    assert ("TTFA", "leading_silence", 2.0, "milliseconds", "component") in values
    assert ("WER", "insertions", 1.0, "percent", "component") in values


def test_failed_or_partial_ttfa_components_are_rejected() -> None:
    skipped: Counter[str] = Counter()
    assert not _metrics_are_valid(
        [
            _row("TTFA", 12.0),
            _row("TTFARoundtrip", 10.0, status="failed"),
            _row("TTFALeadingSilence", 2.0),
        ],
        skipped,
    )
    assert skipped["metric_payload_invalid"] == 1


@pytest.mark.parametrize("anchor_status", ["success", "failed"])
def test_ttfa_components_require_a_successful_valued_anchor(anchor_status: str) -> None:
    anchor = _row("TTFA", None, status=anchor_status)
    if anchor_status == "failed":
        anchor = replace(anchor, error="no TTFA produced")
    skipped: Counter[str] = Counter()
    assert not _metrics_are_valid(
        [anchor, _row("TTFARoundtrip", 10.0), _row("TTFALeadingSilence", 2.0)],
        skipped,
    )
    assert skipped["metric_payload_invalid"] == 1


def test_tts_failed_anchor_without_filename_requires_live_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    skipped: Counter[str] = Counter()
    plans = _tts_plans([_row("TTFA", 12.0, filename=None, status="failed")], skipped)
    assert len(plans) == 1
    assert plans[0].live_owner_required
    assert plans[0].source_kind == "generated_audio"
    assert plans[0].artifacts == [("timing_events", '{"ttfa_ms":12.0}')]
    assert skipped == {}


def test_tts_failed_prompt_is_independent_of_other_prompt_in_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    first = _row("TTFA", 12.0, filename=None, status="failed", transcript="first")
    second = replace(_row("TTFA", 13.0, transcript="second"), id=2)
    plans = _tts_plans([first, second], Counter())
    assert len(plans) == 2
    assert plans[0].live_owner_required
    assert not plans[1].live_owner_required


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        (None, "milliseconds"),
        (float("nan"), "milliseconds"),
        (float("inf"), "milliseconds"),
        (-1.0, "milliseconds"),
        (12.0, "seconds"),
    ],
)
def test_tts_filename_less_allowlist_rejects_malformed_timing(
    monkeypatch: pytest.MonkeyPatch, value: float | None, unit: str
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    row = replace(_row("TTFA", value, filename=None, status="failed"), unit=unit)
    skipped: Counter[str] = Counter()
    assert _tts_plans([row], skipped) == []
    assert skipped == {"tts_anchor_filename_missing": 1, "tts_source_rows_unclaimed": 1}


def test_tts_success_without_filename_remains_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    skipped: Counter[str] = Counter()
    assert _tts_plans([_row("TTFA", 12.0, filename=None)], skipped) == []
    assert skipped == {"tts_anchor_filename_missing": 1, "tts_source_rows_unclaimed": 1}


def test_tts_failed_empty_filename_is_not_the_current_null_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    skipped: Counter[str] = Counter()
    row = _row("TTFA", 12.0, filename="", status="failed")
    assert _tts_plans([row], skipped) == []
    assert skipped == {"tts_anchor_filename_missing": 1, "tts_source_rows_unclaimed": 1}


def test_tts_failed_error_is_preserved_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    row = replace(
        _row("TTFA", 12.0, filename=None, status="failed"),
        error=" provider failed ",
    )
    plan = _tts_plans([row], Counter())[0]
    assert plan.error == " provider failed "
    cursor = _LiveCursor(plan)
    assert _validate_live(cursor, plan).matches
    assert cursor.evaluations[0][9] == " provider failed "


def test_canonical_missing_ttfa_is_metric_failure_not_fabricated_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    anchor = replace(_row("TTFA", None, status="failed"), error="no TTFA produced")
    skipped: Counter[str] = Counter()
    plans = _tts_plans([anchor], skipped)
    assert len(plans) == 1
    assert plans[0].status == "succeeded"
    assert plans[0].error is None
    assert skipped == {}


def test_ambiguous_tts_provider_failure_origin_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    anchor = replace(_row("TTFA", None, status="failed"), error="upstream exploded")
    skipped: Counter[str] = Counter()
    assert _tts_plans([anchor], skipped) == []
    assert skipped["observation_failure_origin_unrecoverable"] == 1


def test_cli_apply_requires_explicit_frozen_window_and_maximum() -> None:
    result = CliRunner().invoke(backfill_normalized_storage_cli, ["--apply"])
    assert result.exit_code == 2
    assert (
        "--apply requires explicit --window-start, --window-end, and --max-result-id"
        in result.output
    )


@pytest.mark.parametrize(
    "missing",
    ["--window-start", "--window-end", "--max-result-id"],
)
def test_cli_apply_rejects_each_missing_frozen_argument_before_connecting(missing: str) -> None:
    arguments = {
        "--window-start": "2026-08-20T00:00:00Z",
        "--window-end": "2026-08-27T00:00:00Z",
        "--max-result-id": "9",
    }
    command = ["--apply"] + [
        item for pair in arguments.items() if pair[0] != missing for item in pair
    ]
    result = CliRunner().invoke(backfill_normalized_storage_cli, command)
    assert result.exit_code == 2
    assert "--apply requires explicit" in result.output


def test_frozen_window_is_utc_and_exactly_seven_days() -> None:
    start = datetime(2026, 8, 20, tzinfo=UTC)
    window = migration.FrozenWindow(1, 9, start, start + timedelta(hours=168))
    assert window.start == start
    assert window.end == start + timedelta(days=7)
    with pytest.raises(ValueError, match="exactly 168 hours"):
        migration.FrozenWindow(1, 9, start, start + timedelta(days=7, seconds=1))


@pytest.mark.parametrize("reason", sorted(migration._LIVE_CONFLICT_CATEGORIES))
def test_parity_mismatch_report_aggregates_safe_reasons(reason: str) -> None:
    report = migration._report()
    migration._append_mismatch(
        report,
        "parity_mismatches",
        {"reason": reason, "natural_key": ["private"]},
    )
    assert report["parity_mismatch_count"] == 1
    assert report["parity_mismatch_reasons"] == {reason: 1}
    assert report["parity_mismatches"] == [{"reason": reason}]


def test_apply_core_requires_explicit_window_before_client_or_connection_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gcs_storage, "Client", lambda: pytest.fail("must not create GCS client"))

    class Conn:
        def cursor(self) -> Any:
            pytest.fail("must not touch the connection")

    with pytest.raises(ValueError, match="apply requires explicit window"):
        migration.backfill(
            Conn(),  # type: ignore[arg-type]
            min_result_id=1,
            max_result_id=1,
            batch_size=1,
            apply=True,
            artifact_bucket="bucket",
        )


def test_stt_tts_rollup_sql_never_deletes_or_verifies_s2s_rows() -> None:
    assert "benchmark IN ('STT','TTS')" in migration._ROLLUP_PAYLOAD_SQL
    assert "benchmark IN ('STT','TTS')" in migration._STORED_ROLLUP_SQL

    class Cursor:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str, _: Any) -> None:
            self.statements.append(statement)

    cursor = Cursor()
    migration._refresh_bucket(cursor, _NOW)  # type: ignore[arg-type]
    assert "benchmark IN ('STT','TTS')" in cursor.statements[1]


def test_backfill_owned_row_keeps_strict_lifecycle_timestamp_comparison() -> None:
    row = replace(_row("WER", 6.0), benchmark="STT")
    plan = migration.Planned(
        [row],
        "sample",
        row.dataset_id,
        row.dataset_sha256,
        "STT",
        "dataset_audio",
        "succeeded",
        None,
        None,
        [("provider_transcript", row.transcript or "")],
    )
    parent = (
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
    later = row.created_at + timedelta(seconds=1)
    stored_values = [
        (
            metric,
            "v1",
            "default",
            "inline",
            "succeeded",
            later,
            later,
            None,
            key,
            unit,
            value,
            role,
        )
        for metric, key, value, unit, role in migration._values(plan.rows)
    ]

    class Cursor:
        def execute(self, *_: Any) -> None:
            return None

        def fetchone(self) -> tuple[Any, ...]:
            return parent

        def fetchall(self) -> list[tuple[Any, ...]]:
            return stored_values

    assert not migration._stored_plan_matches(Cursor(), plan, plan.id)  # type: ignore[arg-type]


class _LiveCursor:
    def __init__(self, plan: migration.Planned) -> None:
        row = plan.first
        self.observation_id = uuid4()
        self.evaluation_id = uuid4()
        self.timing_id = uuid4()
        self.audio_id = uuid4()
        self.parent: tuple[Any, ...] = (
            row.run_id,
            plan.dataset_id,
            plan.dataset_sha256,
            plan.sample_id,
            row.provider,
            row.model,
            row.voice,
            plan.benchmark,
            plan.source_kind,
            None,
            None,
            None,
            row.created_at,
            plan.status,
            plan.error,
            plan.failure_origin,
        )
        expected_metrics = migration._expected_metric_payloads(plan)
        self.evaluations: list[tuple[Any, ...]] = [
            (
                self.evaluation_id,
                metric,
                "v1",
                "default",
                "inline",
                None,
                status,
                row.created_at,
                row.created_at,
                error,
            )
            for metric, (status, error, _) in expected_metrics.items()
        ]
        self.values: list[tuple[Any, ...]] = [
            (metric, key, unit, value, role)
            for metric, (_, _, values) in expected_metrics.items()
            for _, key, value, unit, role in values
        ]
        _, timing_payload, _, _, _ = prepare_timing_events({"ttfa_ms": row.value})
        timing_digest = hashlib.sha256(timing_payload).hexdigest()
        self.artifacts: list[tuple[Any, ...]] = [
            (
                self.timing_id,
                "timing_events",
                "TimingEvents",
                "v1",
                f"gs://private/observation-artifacts/v1/timing_events/{timing_digest[:2]}/{timing_digest}.json",
                timing_digest,
                len(timing_payload),
                None,
            )
        ]
        if not plan.live_owner_required:
            audio_digest = "b" * 64
            self.artifacts.append(
                (
                    self.audio_id,
                    "generated_audio",
                    "GeneratedAudio",
                    "v1",
                    f"gs://private/observation-artifacts/v1/generated_audio/bb/{audio_digest}.wav",
                    audio_digest,
                    10,
                    1.0,
                )
            )
        self.inputs: list[tuple[Any, ...]] = [
            ("TTFA", "timing", 0, self.timing_id),
            *([] if plan.live_owner_required else [("TTFA", "raw", 0, self.audio_id)]),
        ]
        self.unexpected = False
        self._result: Any = None

    def execute(self, statement: str, *_: Any) -> None:
        if "FROM benchmarks_v2.benchmark_observations WHERE id" in statement:
            self._result = self.parent
        elif "FROM benchmarks_v2.metric_evaluations WHERE observation_id" in statement:
            self._result = self.evaluations
        elif "FROM benchmarks_v2.metric_values v JOIN" in statement:
            self._result = self.values
        elif "FROM benchmarks_v2.observation_artifacts WHERE observation_id" in statement:
            self._result = self.artifacts
        elif "FROM benchmarks_v2.metric_evaluation_inputs i JOIN" in statement:
            self._result = self.inputs
        elif "SELECT EXISTS" in statement:
            self._result = (self.unexpected,)
        else:  # pragma: no cover - catches validator query drift.
            raise AssertionError(statement)

    def fetchone(self) -> tuple[Any, ...] | None:
        return cast(tuple[Any, ...] | None, self._result)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return cast(list[tuple[Any, ...]], self._result)


def _validate_live(cursor: _LiveCursor, plan: migration.Planned) -> migration.LiveValidationResult:
    return migration._live_validation_result(
        cast(psycopg.Cursor[tuple[Any, ...]], cursor), plan, cursor.observation_id
    )


def _live_tts_plan(*, failed: bool = False, transport: bool = False) -> migration.Planned:
    row = _row(
        "TTFA",
        12.0,
        status="failed" if failed else "success",
        filename=None if failed else "a.wav",
    )
    if transport:
        row = replace(row, http_version="HTTP/2", headers_ms=4.0)
    return migration.Planned(
        [row],
        "sample",
        row.dataset_id,
        row.dataset_sha256,
        "TTS",
        "generated_audio",
        "failed" if failed else "succeeded",
        row.error if failed else None,
        "provider" if failed else None,
        [("timing_events", '{"ttfa_ms":12.0}')],
        failed,
    )


def test_live_timing_only_failed_tts_matches_exact_current_shape() -> None:
    plan = _live_tts_plan(failed=True)
    cursor = _LiveCursor(plan)
    result = _validate_live(cursor, plan)
    assert result.matches
    assert result.category is None
    assert cursor.values == []
    assert [artifact[1] for artifact in cursor.artifacts] == ["timing_events"]
    assert cursor.inputs == [("TTFA", "timing", 0, cursor.timing_id)]


def test_live_validation_reports_exact_aggregate_safe_categories() -> None:
    plan = _live_tts_plan()

    parent = _LiveCursor(plan)
    parent.parent = (*parent.parent[:4], "foreign", *parent.parent[5:])
    assert _validate_live(parent, plan).category == "parent"

    lifecycle = _LiveCursor(plan)
    lifecycle.parent = (*lifecycle.parent[:13], "failed", "mutated", "provider")
    assert _validate_live(lifecycle, plan).category == "lifecycle"

    evaluation = _LiveCursor(plan)
    evaluation.evaluations[0] = (
        *evaluation.evaluations[0][:2],
        "v2",
        *evaluation.evaluations[0][3:],
    )
    assert _validate_live(evaluation, plan).category == "evaluation"

    value = _LiveCursor(plan)
    value.values[0] = (*value.values[0][:3], 99.0, value.values[0][4])
    assert _validate_live(value, plan).category == "value"

    artifact = _LiveCursor(plan)
    artifact.artifacts[0] = (*artifact.artifacts[0][:5], "c" * 64, *artifact.artifacts[0][6:])
    assert _validate_live(artifact, plan).category == "artifact"

    input_mismatch = _LiveCursor(plan)
    input_mismatch.inputs[0] = ("TTFA", "raw", 0, input_mismatch.timing_id)
    assert _validate_live(input_mismatch, plan).category == "input"

    foreign_input = _LiveCursor(plan)
    foreign_input.inputs[0] = ("TTFA", "timing", 0, uuid4())
    assert _validate_live(foreign_input, plan).category == "input"

    duplicate_evaluation = _LiveCursor(plan)
    duplicate_evaluation.evaluations.append(duplicate_evaluation.evaluations[0])
    assert _validate_live(duplicate_evaluation, plan).category == "unexpected_child"

    duplicate_artifact = _LiveCursor(plan)
    duplicate_artifact.artifacts.append(duplicate_artifact.artifacts[0])
    assert _validate_live(duplicate_artifact, plan).category == "unexpected_child"

    extra_child_table = _LiveCursor(plan)
    extra_child_table.unexpected = True
    assert _validate_live(extra_child_table, plan).category == "unexpected_child"


def test_live_validation_keeps_successful_tts_generated_audio_strict() -> None:
    plan = _live_tts_plan()
    missing = _LiveCursor(plan)
    missing.artifacts = missing.artifacts[:1]
    assert _validate_live(missing, plan).category == "artifact"

    malformed = _LiveCursor(plan)
    malformed.artifacts[1] = (*malformed.artifacts[1][:-1], 0.0)
    assert _validate_live(malformed, plan).category == "artifact"


def test_live_validation_timestamps_are_semantic_and_provenance_is_actual() -> None:
    plan = _live_tts_plan(transport=True)
    exact = _LiveCursor(plan)
    first = _validate_live(exact, plan)
    second = _validate_live(_LiveCursor(plan), plan)
    assert first == second
    assert first.matches
    assert first.tolerated_provenance == (
        "artifact_content_not_reconstructable",
        "legacy_transport_metadata",
    )

    semantic = _LiveCursor(plan)
    later = plan.first.created_at + timedelta(seconds=1)
    semantic.evaluations[0] = (
        *semantic.evaluations[0][:7],
        later,
        later + timedelta(seconds=1),
        semantic.evaluations[0][9],
    )
    result = _validate_live(semantic, plan)
    assert result.matches
    assert "live_evaluation_timestamps" in result.tolerated_provenance

    invalid = _LiveCursor(plan)
    invalid.evaluations[0] = (
        *invalid.evaluations[0][:7],
        plan.first.created_at - timedelta(seconds=1),
        plan.first.created_at,
        invalid.evaluations[0][9],
    )
    assert _validate_live(invalid, plan).category == "evaluation"


def test_live_validation_rejects_stored_transport_mutation() -> None:
    plan = _live_tts_plan()
    cursor = _LiveCursor(plan)
    cursor.parent = (*cursor.parent[:9], "HTTP/2", *cursor.parent[10:])
    assert _validate_live(cursor, plan).category == "parent"


def test_timing_only_failed_tts_without_live_owner_never_inserts() -> None:
    plan = _live_tts_plan(failed=True)
    statements: list[str] = []

    class Cursor:
        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def execute(self, statement: str, *_: Any) -> None:
            statements.append(statement)

        def fetchone(self) -> None:
            return None

    class Conn:
        def cursor(self) -> Cursor:
            return Cursor()

    report = migration._report()
    migration._insert_plan(Conn(), plan, True, "unused", object(), report)  # type: ignore[arg-type]
    assert not any("INSERT" in statement for statement in statements)
    assert report["eligible"] == 0
    assert report["parity_mismatch_reasons"] == {"parent": 1}


def test_timing_only_failed_tts_deterministic_owner_is_not_live_proof() -> None:
    plan = _live_tts_plan(failed=True)
    statements: list[str] = []

    class Cursor:
        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def execute(self, statement: str, *_: Any) -> None:
            statements.append(statement)

        def fetchone(self) -> tuple[Any, ...]:
            return (plan.id,)

    class Conn:
        def cursor(self) -> Cursor:
            return Cursor()

    report = migration._report()
    migration._insert_plan(Conn(), plan, False, None, None, report)  # type: ignore[arg-type]
    assert len(statements) == 1
    assert report["reconciled"] == 0
    assert report["parity_mismatch_reasons"] == {"parent": 1}


def test_live_tts_artifacts_inputs_and_lifecycle_are_exact_where_recoverable() -> None:
    row = _row("TTFA", 12.0)
    plan = migration.Planned(
        [row],
        "sample",
        row.dataset_id,
        row.dataset_sha256,
        "TTS",
        "generated_audio",
        "succeeded",
        None,
        None,
        [("timing_events", '{"ttfa_ms":12.0}')],
    )
    observation_id, timing_id, audio_id = uuid4(), uuid4(), uuid4()
    _, timing_payload, _, _, _ = prepare_timing_events({"ttfa_ms": 12.0})
    timing_digest = hashlib.sha256(timing_payload).hexdigest()
    audio_digest = "b" * 64

    def matches(
        *,
        timing: tuple[str, int],
        started: datetime = _NOW + timedelta(seconds=1),
        audio_duration: float = 1.0,
    ) -> bool:
        parent = (
            row.run_id,
            plan.dataset_id,
            plan.dataset_sha256,
            plan.sample_id,
            row.provider,
            row.model,
            row.voice,
            "TTS",
            "generated_audio",
            None,
            None,
            None,
            _NOW,
            "succeeded",
            None,
            None,
        )
        artifacts = [
            (
                timing_id,
                "timing_events",
                "TimingEvents",
                "v1",
                f"gs://private/observation-artifacts/v1/timing_events/{timing[0][:2]}/{timing[0]}.json",
                timing[0],
                timing[1],
                None,
            ),
            (
                audio_id,
                "generated_audio",
                "GeneratedAudio",
                "v1",
                f"gs://private/observation-artifacts/v1/generated_audio/bb/{audio_digest}.wav",
                audio_digest,
                10,
                audio_duration,
            ),
        ]

        class Cursor:
            fetchone_calls = 0
            fetchall_calls = 0

            def execute(self, *_: Any) -> None:
                return None

            def fetchone(self) -> tuple[Any, ...]:
                self.fetchone_calls += 1
                return parent if self.fetchone_calls == 1 else (False,)

            def fetchall(self) -> list[tuple[Any, ...]]:
                self.fetchall_calls += 1
                return cast(
                    list[tuple[Any, ...]],
                    {
                        1: [
                            (
                                uuid4(),
                                "TTFA",
                                "v1",
                                "default",
                                "inline",
                                None,
                                "succeeded",
                                started,
                                started + timedelta(seconds=1),
                                None,
                            )
                        ],
                        2: [("TTFA", "primary", "milliseconds", 12.0, "primary")],
                        3: artifacts,
                        4: [("TTFA", "timing", 0, timing_id), ("TTFA", "raw", 0, audio_id)],
                    }[self.fetchall_calls],
                )

        return migration._live_stored_plan_matches(Cursor(), plan, observation_id)  # type: ignore[arg-type]

    assert matches(timing=(timing_digest, len(timing_payload)))
    assert not matches(timing=("c" * 64, 9))
    assert not matches(
        timing=(timing_digest, len(timing_payload)), started=_NOW - timedelta(seconds=1)
    )
    assert not matches(timing=(timing_digest, len(timing_payload)), audio_duration=0)


def test_live_tts_null_ttfa_keeps_unreferenced_timing_artifact_for_wer() -> None:
    anchor = _row("TTFA", None)
    wer = _row("WER", 6.0)
    plan = migration.Planned(
        [anchor, wer],
        "sample",
        anchor.dataset_id,
        anchor.dataset_sha256,
        "TTS",
        "generated_audio",
        "succeeded",
        None,
        None,
        [],
    )
    observation_id, timing_id, audio_id = uuid4(), uuid4(), uuid4()
    # The legacy transport-contaminated row is null; the private artifact kept
    # the unrecoverable original provider timing.
    _, timing_payload, _, _, _ = prepare_timing_events({"ttfa_ms": 47.25})
    timing_digest = hashlib.sha256(timing_payload).hexdigest()
    audio_digest = "b" * 64
    parent = (
        anchor.run_id,
        plan.dataset_id,
        plan.dataset_sha256,
        plan.sample_id,
        anchor.provider,
        anchor.model,
        anchor.voice,
        "TTS",
        "generated_audio",
        None,
        None,
        None,
        _NOW,
        "succeeded",
        None,
        None,
    )

    class Cursor:
        fetchone_calls = 0
        fetchall_calls = 0

        def execute(self, *_: Any) -> None:
            return None

        def fetchone(self) -> tuple[Any, ...]:
            self.fetchone_calls += 1
            return parent if self.fetchone_calls == 1 else (False,)

        def fetchall(self) -> list[tuple[Any, ...]]:
            self.fetchall_calls += 1
            return cast(
                list[tuple[Any, ...]],
                {
                    1: [
                        (
                            uuid4(),
                            "WER",
                            "v1",
                            "default",
                            "inline",
                            None,
                            "succeeded",
                            _NOW,
                            _NOW,
                            None,
                        )
                    ],
                    2: [
                        ("WER", key, unit, value, role)
                        for _, key, value, unit, role in migration._values(plan.rows)
                    ],
                    3: [
                        (
                            timing_id,
                            "timing_events",
                            "TimingEvents",
                            "v1",
                            f"gs://private/observation-artifacts/v1/timing_events/{timing_digest[:2]}/{timing_digest}.json",
                            timing_digest,
                            len(timing_payload),
                            None,
                        ),
                        (
                            audio_id,
                            "generated_audio",
                            "GeneratedAudio",
                            "v1",
                            f"gs://private/observation-artifacts/v1/generated_audio/bb/{audio_digest}.wav",
                            audio_digest,
                            10,
                            1.0,
                        ),
                    ],
                    4: [("WER", "raw", 0, audio_id)],
                }[self.fetchall_calls],
            )

    assert migration._live_stored_plan_matches(Cursor(), plan, observation_id)  # type: ignore[arg-type]


def test_live_stt_wer_allows_required_unreferenced_timing_artifact() -> None:
    row = replace(_row("WER", 6.0), benchmark="STT")
    plan = migration.Planned(
        [row],
        "sample",
        row.dataset_id,
        row.dataset_sha256,
        "STT",
        "dataset_audio",
        "succeeded",
        None,
        None,
        [("provider_transcript", row.transcript or "")],
    )
    observation_id, transcript_id, timing_id = uuid4(), uuid4(), uuid4()
    _, transcript_payload, _, _, _ = prepare_provider_transcript(row.transcript or "")
    transcript_digest = hashlib.sha256(transcript_payload).hexdigest()
    timing_digest = "c" * 64
    parent = (
        row.run_id,
        plan.dataset_id,
        plan.dataset_sha256,
        plan.sample_id,
        row.provider,
        row.model,
        row.voice,
        "STT",
        "dataset_audio",
        None,
        None,
        None,
        _NOW,
        "succeeded",
        None,
        None,
    )

    class Cursor:
        fetchone_calls = 0
        fetchall_calls = 0

        def execute(self, *_: Any) -> None:
            return None

        def fetchone(self) -> tuple[Any, ...]:
            self.fetchone_calls += 1
            return parent if self.fetchone_calls == 1 else (False,)

        def fetchall(self) -> list[tuple[Any, ...]]:
            self.fetchall_calls += 1
            return cast(
                list[tuple[Any, ...]],
                {
                    1: [
                        (
                            uuid4(),
                            "WER",
                            "v1",
                            "default",
                            "inline",
                            None,
                            "succeeded",
                            _NOW,
                            _NOW,
                            None,
                        )
                    ],
                    2: [
                        ("WER", key, unit, value, role)
                        for _, key, value, unit, role in migration._values(plan.rows)
                    ],
                    3: [
                        (
                            transcript_id,
                            "provider_transcript",
                            "ProviderTranscript",
                            "v1",
                            f"gs://private/observation-artifacts/v1/provider_transcript/{transcript_digest[:2]}/{transcript_digest}.json",
                            transcript_digest,
                            len(transcript_payload),
                            None,
                        ),
                        (
                            timing_id,
                            "timing_events",
                            "TimingEvents",
                            "v1",
                            f"gs://private/observation-artifacts/v1/timing_events/{timing_digest[:2]}/{timing_digest}.json",
                            timing_digest,
                            1,
                            None,
                        ),
                    ],
                    4: [("WER", "raw", 0, transcript_id)],
                }[self.fetchall_calls],
            )

    assert migration._live_stored_plan_matches(Cursor(), plan, observation_id)  # type: ignore[arg-type]


def test_cli_rejects_the_local_placeholder_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Settings:
        database_url = "postgresql://unused:unused@127.0.0.1:5432/unused"

    monkeypatch.setattr(migration, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        psycopg, "connect", lambda *_: pytest.fail("must not connect to placeholder")
    )
    result = CliRunner().invoke(backfill_normalized_storage_cli, ["--max-result-id", "1"])
    assert result.exit_code == 1
    assert "DATABASE_URL is required" in result.output


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _progress_events(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_progress_reporter_emits_periodically_with_checkpoint_throughput_and_eta() -> None:
    clock = _Clock()
    stream = io.StringIO()
    report = migration._report()
    reporter = migration.ProgressReporter(
        "dry_run", 1, 99, 100, stream=stream, monotonic=clock, max_units=10
    )
    reporter.total_runs = 4
    reporter.phase_started("source_reconciliation", report, total_runs=4)
    clock.value = 31
    row = replace(_row("WER", 1.0), id=9, run_id=7, benchmark="STT")
    reporter.completed_page([7], [row], report, phase="source_reconciliation")

    event = _progress_events(stream)[-1]
    assert event["status"] == "progress"
    assert event["last_completed_run_id"] == 7
    assert event["last_completed_result_id"] == 9
    assert event["throughput_runs_per_second"] == pytest.approx(1 / 31, rel=1e-5)
    assert event["throughput_results_per_second"] == pytest.approx(1 / 31, rel=1e-5)
    assert event["eta_seconds"] == pytest.approx(93)
    assert event["phase_elapsed_seconds"] == 31
    assert event["mode"] == "dry_run"


def test_progress_reporter_uses_bounded_unit_fallback_before_time_interval() -> None:
    clock = _Clock()
    stream = io.StringIO()
    report = migration._report()
    reporter = migration.ProgressReporter(
        "dry_run", 1, 99, 100, stream=stream, monotonic=clock, max_units=2
    )
    reporter.phase_started("source_reconciliation", report)
    row = replace(_row("WER", 1.0), benchmark="STT")
    reporter.completed_page([1], [row], report, phase="source_reconciliation")
    reporter.completed_page(
        [2], [replace(row, id=2, run_id=2)], report, phase="source_reconciliation"
    )

    assert _progress_events(stream)[-1]["status"] == "progress"


def test_progress_reporter_tracks_verification_pages_and_rollup_units() -> None:
    clock = _Clock()
    stream = io.StringIO()
    report = migration._report()
    reporter = migration.ProgressReporter("apply", 1, 99, 100, stream=stream, monotonic=clock)
    row = replace(_row("WER", 1.0), id=9, run_id=7, benchmark="STT")

    reporter.phase_started("post_write_verification", report, total_runs=4)
    clock.value = 10
    reporter.completed_page([7], [row], report, phase="post_write_verification")
    reporter.phase_completed("post_write_verification", report)
    verification = _progress_events(stream)[-1]
    assert verification["phase_pages"] == 1
    assert verification["phase_runs"] == 1
    assert verification["phase_results"] == 1
    assert verification["eta_seconds"] == pytest.approx(30)

    reporter.phase_started("rollup_verification", report)
    reporter.completed_unit(report, phase="rollup_verification")
    reporter.phase_completed("rollup_verification", report)
    rollup = _progress_events(stream)[-1]
    assert rollup["phase_units_completed"] == 1
    assert "eta_seconds" not in rollup


def test_dry_run_progress_is_stderr_only_and_forces_phase_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = replace(_row("WER", 1.0), id=9, run_id=7, benchmark="STT", transcript="private")
    stream = io.StringIO()
    reporter = migration.ProgressReporter("dry_run", 1, 9, 1, stream=stream, monotonic=_Clock())
    writes: list[bool] = []
    plan = migration.Planned(
        [row],
        "sample",
        row.dataset_id,
        row.dataset_sha256,
        "STT",
        "dataset_audio",
        "succeeded",
        None,
        None,
        [],
    )

    class Conn:
        def commit(self) -> None:
            return None

    monkeypatch.setattr(migration, "_qualifying_run_count", lambda *_: 1)
    monkeypatch.setattr(migration, "_complete_pages", lambda *_: iter([([row], [row])]))
    monkeypatch.setattr(migration, "_page_plans", lambda *_: [plan])
    monkeypatch.setattr(migration, "_scheduled_buckets", lambda *_: iter(()))
    monkeypatch.setattr(migration, "_rollup_mismatches", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        migration, "_insert_plan", lambda _conn, _plan, apply, *_args: writes.append(apply)
    )

    report = migration.backfill(
        Conn(),  # type: ignore[arg-type]
        min_result_id=1,
        max_result_id=9,
        batch_size=1,
        apply=False,
        reporter=reporter,
    )

    events = _progress_events(stream)
    assert writes == [False]
    assert report["source_rows"] == 1
    assert events[-1]["phase"] == "operation"
    assert events[-1]["status"] == "completed"
    assert "private" not in stream.getvalue()


def test_cli_keeps_progress_on_stderr_and_machine_report_as_final_stdout_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Settings:
        database_url = "postgresql://example"

    class Conn:
        def __enter__(self) -> Conn:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def commit(self) -> None:
            return None

    row = replace(_row("WER", 1.0), id=9, run_id=7, benchmark="STT")
    monkeypatch.setattr(migration, "get_settings", lambda: Settings())
    monkeypatch.setattr(psycopg, "connect", lambda _: Conn())
    monkeypatch.setattr(migration, "_qualifying_run_count", lambda *_: 1)
    monkeypatch.setattr(migration, "_complete_pages", lambda *_: iter([([row], [row])]))
    monkeypatch.setattr(migration, "_page_plans", lambda *_: [])
    monkeypatch.setattr(migration, "_scheduled_buckets", lambda *_: iter(()))
    monkeypatch.setattr(migration, "_rollup_mismatches", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        backfill_normalized_storage_cli,
        [
            "--max-result-id",
            "9",
            "--window-start",
            "2026-08-20T00:00:00Z",
            "--window-end",
            "2026-08-27T00:00:00Z",
        ],
    )

    assert result.exit_code == 0
    assert "normalized_storage_backfill_progress" not in result.stdout
    assert "normalized_storage_backfill_progress" in result.stderr
    window = json.loads(result.stdout.splitlines()[-1])["window"]
    assert window == {
        "batch_size": 100,
        "end": "2026-08-27T00:00:00+00:00",
        "max_result_id": 9,
        "min_result_id": 1,
        "start": "2026-08-20T00:00:00+00:00",
    }


def test_cli_dry_run_freezes_database_clock_and_maximum_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_now = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    captured: dict[str, Any] = {}

    class Settings:
        database_url = "postgresql://example"

    class Cursor:
        calls = 0

        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def execute(self, statement: str) -> None:
            assert "statement_timestamp" in statement
            self.calls += 1

        def fetchone(self) -> tuple[datetime, int]:
            return database_now, 77

    class Conn:
        cursor_value = Cursor()

        def __enter__(self) -> Conn:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def cursor(self) -> Cursor:
            return self.cursor_value

    def fake_backfill(_: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "window": {
                "start": kwargs["window_start"].isoformat(),
                "end": kwargs["window_end"].isoformat(),
                "max_result_id": kwargs["max_result_id"],
            }
        }

    monkeypatch.setattr(migration, "get_settings", lambda: Settings())
    monkeypatch.setattr(psycopg, "connect", lambda _: Conn())
    monkeypatch.setattr(migration, "backfill", fake_backfill)
    result = CliRunner().invoke(backfill_normalized_storage_cli, [])

    assert result.exit_code == 0
    assert Conn.cursor_value.calls == 1
    assert captured["window_start"] == database_now - timedelta(hours=168)
    assert captured["window_end"] == database_now
    assert captured["max_result_id"] == 77
    assert json.loads(result.stdout.splitlines()[-1])["window"] == {
        "start": "2026-08-20T12:00:00+00:00",
        "end": "2026-08-27T12:00:00+00:00",
        "max_result_id": 77,
    }


def test_sigterm_cancellation_handler_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = object()
    handlers: list[Any] = []
    monkeypatch.setattr(signal, "getsignal", lambda _: previous)
    monkeypatch.setattr(signal, "signal", lambda _, handler: handlers.append(handler))

    with migration._temporary_sigterm_cancellation(), pytest.raises(migration._BackfillCancelled):
        handlers[-1](signal.SIGTERM, None)

    assert handlers[-1] is previous


def test_progress_failure_keeps_last_completed_checkpoint_without_detail_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = replace(_row("WER", 1.0), id=9, run_id=7, benchmark="STT")
    stream = io.StringIO()
    reporter = migration.ProgressReporter("dry_run", 1, 9, 1, stream=stream, monotonic=_Clock())

    def pages(*_: Any) -> Any:
        yield [row], [row]
        raise RuntimeError("private transcript and exception detail")

    class Conn:
        def commit(self) -> None:
            return None

    monkeypatch.setattr(migration, "_qualifying_run_count", lambda *_: 1)
    monkeypatch.setattr(migration, "_complete_pages", pages)
    monkeypatch.setattr(migration, "_page_plans", lambda *_: [])
    with pytest.raises(RuntimeError, match="private transcript"):
        migration.backfill(
            Conn(),  # type: ignore[arg-type]
            min_result_id=1,
            max_result_id=9,
            window_start=_WINDOW_START,
            window_end=_WINDOW_END,
            batch_size=1,
            apply=False,
            reporter=reporter,
        )

    events = _progress_events(stream)
    assert events[-1]["status"] == "failed"
    assert events[-1]["last_completed_run_id"] == 7
    assert events[-1]["last_completed_result_id"] == 9
    assert "private" not in stream.getvalue()


def test_apply_cancellation_emits_checkpoint_then_unlocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = replace(_row("WER", 1.0), id=9, run_id=7, benchmark="STT")
    stream = io.StringIO()
    reporter = migration.ProgressReporter("apply", 1, 9, 1, stream=stream, monotonic=_Clock())
    events: list[str] = []

    def pages(*_: Any) -> Any:
        yield [row], []
        raise migration._BackfillCancelled()

    class Cursor:
        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def execute(self, statement: str, *_: Any) -> None:
            events.append("unlock" if "pg_advisory_unlock" in statement else "lock")

    class Conn:
        def commit(self) -> None:
            events.append("commit")

        def rollback(self) -> None:
            events.append("rollback")

        def cursor(self) -> Cursor:
            return Cursor()

        def transaction(self) -> Any:
            return nullcontext()

    monkeypatch.setattr(migration, "_qualifying_run_count", lambda *_: 1)
    monkeypatch.setattr(migration, "_complete_pages", pages)
    monkeypatch.setattr(migration, "_page_plans", lambda *_: [])
    monkeypatch.setattr(gcs_storage, "Client", lambda: object())
    monkeypatch.setattr(migration, "_preflight_artifact_bucket", lambda *_: None)
    with pytest.raises(migration._BackfillCancelled):
        migration.backfill(
            Conn(),  # type: ignore[arg-type]
            min_result_id=1,
            max_result_id=9,
            window_start=_WINDOW_START,
            window_end=_WINDOW_END,
            batch_size=1,
            apply=True,
            artifact_bucket="bucket",
            reporter=reporter,
        )

    progress = _progress_events(stream)
    assert progress[-1]["status"] == "cancelled"
    assert progress[-1]["last_completed_run_id"] == 7
    assert events[-3:] == ["rollback", "unlock", "commit"]


def test_run_pages_keyset_by_run_id_not_interleaved_result_ids() -> None:
    first = replace(_row("WER", 1.0), id=1, run_id=10, benchmark="STT")
    second = replace(_row("WER", 2.0), id=2, run_id=20, benchmark="STT")

    class Cursor:
        def __init__(self) -> None:
            self.result: list[tuple[Any, ...]] = []
            self.calls: list[tuple[str, tuple[Any, ...]]] = []

        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            self.calls.append((sql, params))
            if sql == migration._PAGE_RUN_IDS_SQL:
                after = int(params[4])
                self.result = [(run_id,) for run_id in (10, 20) if run_id > after][: int(params[5])]
            else:
                self.result = [astuple(first) if params[0] == [10] else astuple(second)]

        def fetchall(self) -> list[tuple[Any, ...]]:
            return self.result

    class Conn:
        def __init__(self) -> None:
            self.value = Cursor()

        def cursor(self) -> Cursor:
            return self.value

    conn = Conn()
    window = migration.FrozenWindow(1, 99, _NOW - timedelta(hours=168), _NOW)
    ids, rows = migration._run_page(conn, window, 0, 1)  # type: ignore[arg-type]
    next_ids, next_rows = migration._run_page(conn, window, ids[-1], 1)  # type: ignore[arg-type]
    assert (ids, [row.run_id for row in rows]) == ([10], [10])
    assert (next_ids, [row.run_id for row in next_rows]) == ([20], [20])
    assert [call[1][4] for call in conn.value.calls if call[0] == migration._PAGE_RUN_IDS_SQL] == [
        0,
        10,
    ]


def test_run_page_uses_start_inclusive_end_exclusive_and_frozen_max_id(
    backfill_pg: psycopg.Connection[Any],
) -> None:
    _migrate(backfill_pg)
    start = datetime(2026, 8, 20, tzinfo=UTC)
    start_id = _insert_wer(backfill_pg, _insert_run(backfill_pg), "start.wav", created_at=start)
    _insert_wer(
        backfill_pg, _insert_run(backfill_pg), "end.wav", created_at=start + timedelta(hours=168)
    )
    max_id = _insert_wer(
        backfill_pg, _insert_run(backfill_pg), "max.wav", created_at=start + timedelta(hours=1)
    )
    _insert_wer(
        backfill_pg,
        _insert_run(backfill_pg),
        "after-max.wav",
        created_at=start + timedelta(hours=1),
    )

    _, rows = migration._run_page(
        backfill_pg,
        migration.FrozenWindow(start_id, max_id, start, start + timedelta(hours=168)),
        0,
        100,
    )
    assert {row.id for row in rows} == {start_id, max_id}


def test_scheduled_buckets_uses_nullable_timestamp_keyset_across_pages() -> None:
    later = _NOW + timedelta(hours=1)

    class Cursor:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []
            self.result: list[tuple[datetime, int]] = []

        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def execute(self, _: str, params: tuple[Any, ...]) -> None:
            self.calls.append(params)
            cursor = params[8]
            self.result = (
                [(_NOW, 10)] if cursor is None else [(later, 20)] if cursor == _NOW else []
            )

        def fetchall(self) -> list[tuple[datetime, int]]:
            return self.result

    class Conn:
        def __init__(self) -> None:
            self.value = Cursor()

        def commit(self) -> None:
            return None

        def cursor(self) -> Cursor:
            return self.value

    conn = Conn()
    window = migration.FrozenWindow(1, 99, _NOW - timedelta(hours=168), _NOW)
    assert list(migration._scheduled_buckets(conn, window, 1)) == [_NOW, later]  # type: ignore[arg-type]
    assert conn.value.calls == [
        (1, 99, window.start, window.end, 1, 99, window.start, window.end, None, None, 0, 1),
        (1, 99, window.start, window.end, 1, 99, window.start, window.end, _NOW, _NOW, 10, 1),
        (1, 99, window.start, window.end, 1, 99, window.start, window.end, later, later, 20, 1),
    ]


def test_apply_replans_for_fresh_verification_without_retaining_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = replace(_row("WER", 1.0), benchmark="STT")
    plan = migration.Planned(
        [row],
        "sample",
        row.dataset_id,
        row.dataset_sha256,
        "STT",
        "dataset_audio",
        "succeeded",
        None,
        None,
        [],
    )
    passes: list[int] = []
    mismatch_flags: list[bool] = []

    def pages(*_: Any) -> Any:
        passes.append(1)
        yield [row], [row]

    class Cursor:
        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def execute(self, *_: Any) -> None:
            return None

        def fetchone(self) -> tuple[Any, ...]:
            return (plan.id,)

    class Conn:
        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

        def cursor(self) -> Cursor:
            return Cursor()

        def transaction(self) -> Any:
            return nullcontext()

    monkeypatch.setattr(migration, "_complete_pages", pages)
    monkeypatch.setattr(migration, "_qualifying_run_count", lambda *_: 1)
    monkeypatch.setattr(migration, "_page_plans", lambda *_: [plan])
    monkeypatch.setattr(gcs_storage, "Client", lambda: object())
    monkeypatch.setattr(migration, "_preflight_artifact_bucket", lambda *_: None)

    def insert(*_args: Any, **kwargs: Any) -> None:
        mismatch_flags.append(kwargs["report_mismatch"])

    monkeypatch.setattr(migration, "_insert_plan", insert)
    monkeypatch.setattr(migration, "_refresh_bucket", lambda *_: None)
    monkeypatch.setattr(migration, "_stored_plan_matches", lambda *_: False)
    monkeypatch.setattr(migration, "_scheduled_buckets", lambda *_: iter(()))
    report = migration.backfill(
        Conn(),  # type: ignore[arg-type]
        min_result_id=1,
        max_result_id=1,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=True,
        artifact_bucket="bucket",
    )
    assert len(passes) == 2
    assert mismatch_flags == [False]
    assert report["parity_mismatch_count"] == 1
    assert report["verification_skipped_by_reason"] == {}


@pytest.mark.parametrize("verification_reason", ["split_window_run", "scheduled_at_missing"])
def test_apply_verification_skips_are_reported_and_block_readiness(
    monkeypatch: pytest.MonkeyPatch, verification_reason: str
) -> None:
    row = replace(_row("WER", 1.0), benchmark="STT")
    verification_row = replace(row, scheduled_at=None)
    plan = migration.Planned(
        [row],
        "sample",
        row.dataset_id,
        row.dataset_sha256,
        "STT",
        "dataset_audio",
        "succeeded",
        None,
        None,
        [],
    )
    verification_plan = replace(plan, rows=[verification_row])
    page_calls = 0

    def pages(
        _conn: Any,
        _window: migration.FrozenWindow,
        _batch_size: int,
        skipped: Counter[str],
    ) -> Any:
        nonlocal page_calls
        page_calls += 1
        if page_calls == 1:
            yield [row], [row]
        elif verification_reason == "split_window_run":
            skipped[verification_reason] += 1
        else:
            yield [verification_row], [verification_row]

    def plans(rows: list[LegacyRow], _: Counter[str]) -> list[migration.Planned]:
        return [verification_plan if rows[0].scheduled_at is None else plan]

    class Cursor:
        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def execute(self, *_: Any) -> None:
            return None

    class Conn:
        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

        def cursor(self) -> Cursor:
            return Cursor()

        def transaction(self) -> Any:
            return nullcontext()

    monkeypatch.setattr(migration, "_complete_pages", pages)
    monkeypatch.setattr(migration, "_qualifying_run_count", lambda *_: 1)
    monkeypatch.setattr(migration, "_page_plans", plans)
    monkeypatch.setattr(gcs_storage, "Client", lambda: object())
    monkeypatch.setattr(migration, "_preflight_artifact_bucket", lambda *_: None)
    monkeypatch.setattr(migration, "_insert_plan", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(migration, "_refresh_bucket", lambda *_: None)
    monkeypatch.setattr(migration, "_scheduled_buckets", lambda *_: iter(()))

    report = migration.backfill(
        Conn(),  # type: ignore[arg-type]
        min_result_id=1,
        max_result_id=1,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=True,
        artifact_bucket="bucket",
    )

    assert report["skipped_by_reason"] == {}
    assert report["verification_skipped_by_reason"] == {verification_reason: 1}
    assert report["parity_mismatch_count"] == 0
    assert not report["cutover_ready"]


def test_dry_run_report_has_no_verification_skips() -> None:
    report = migration._report()
    report["eligible"] = 1
    migration._set_ready(report, dry_run=True)
    assert report["verification_skipped_by_reason"] == {}
    assert not report["cutover_ready"]


def test_apply_bucket_preflight_fails_before_lock_page_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Bucket:
        def test_iam_permissions(self, _: list[str]) -> list[str]:
            events.append("preflight")
            return []

    class Storage:
        def bucket(self, _: str) -> Bucket:
            return Bucket()

    class Conn:
        def commit(self) -> None:
            events.append("commit")

        def cursor(self) -> Any:
            events.append("lock")
            return nullcontext()

    monkeypatch.setattr(gcs_storage, "Client", lambda: Storage())
    with pytest.raises(click.ClickException, match="storage.objects.create"):
        migration.backfill(
            Conn(),  # type: ignore[arg-type]
            min_result_id=1,
            max_result_id=1,
            window_start=_WINDOW_START,
            window_end=_WINDOW_END,
            batch_size=1,
            apply=True,
            artifact_bucket="bucket",
        )
    assert events == ["preflight"]


def test_packaged_manifest_index_preserves_ambiguity_and_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration._packaged_manifest.cache_clear()
    reads = 0

    class Resource:
        def read_bytes(self) -> bytes:
            nonlocal reads
            reads += 1
            return (
                b'{"items":[{"path":"audio/a.wav","sample_id":"sample"},'
                b'{"testcase_id":"T1","transcript":"prompt"},'
                b'{"testcase_id":"T2","transcript":"prompt"}]}'
            )

    class Package:
        def joinpath(self, _: str) -> Resource:
            return Resource()

    monkeypatch.setattr(migration, "files", lambda _: Package())
    index = migration._packaged_manifest("tts-v1")
    assert index is not None
    assert migration._stt_sample("tts-v1", index.sha256, "a.wav") == "sample"
    assert migration._tts_sample("tts-v1", index.sha256, "prompt") is None
    assert reads == 1
    migration._packaged_manifest.cache_clear()


def test_mismatch_details_are_capped_but_counts_remain_exact() -> None:
    report = migration._report()
    for index in range(101):
        migration._append_mismatch(report, "parity_mismatches", {"index": index})
    migration._set_ready(report, dry_run=False)
    assert report["parity_mismatch_count"] == 101
    assert len(report["parity_mismatches"]) == 100
    assert report["parity_mismatches_truncated"]
    assert not report["cutover_ready"]


def test_runner_image_allows_cloud_run_to_override_the_default_command() -> None:
    dockerfile = (Path(__file__).parents[2] / "Dockerfile").read_text()
    assert 'ENTRYPOINT ["python", "-m", "coval_bench"]' in dockerfile
    assert 'CMD ["run"]' in dockerfile


def test_repository_owned_container_overrides_use_the_image_entrypoint() -> None:
    root = Path(__file__).parents[2]
    compose = (root / ".." / "docker-compose.yml").read_text()
    readme = (root / "README.md").read_text()
    arena = (root / ".." / "scripts" / "arena-local.sh").read_text()
    assert 'command: ["db", "migrate"]' in compose
    assert "docker compose run --rm runner run --smoke --kind tts" in compose
    assert "docker compose run --rm runner run --smoke --kind tts" in readme
    assert "docker compose run --rm runner tts-smoke" in readme
    assert "docker compose run --rm runner arena snapshot" in arena


def test_apply_reconciles_artifacts_inputs_values_and_rollups(
    backfill_pg: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _migrate(backfill_pg)
    run_id = _insert_run(backfill_pg)
    result_id = _insert_wer(backfill_pg, run_id, "sample.wav")
    fake_storage = _Storage()
    monkeypatch.setattr(gcs_storage, "Client", lambda: fake_storage)

    dry_run = backfill(
        backfill_pg,
        min_result_id=result_id,
        max_result_id=result_id,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=False,
    )
    assert not dry_run["cutover_ready"]
    assert dry_run["eligible"] == 1
    with backfill_pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.benchmark_observations")
        assert cur.fetchone() == (0,)
    backfill_pg.rollback()

    report = backfill(
        backfill_pg,
        min_result_id=result_id,
        max_result_id=result_id,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=True,
        artifact_bucket="backfill-artifacts",
    )
    assert report["cutover_ready"]
    assert report["rollup_mismatches"] == []
    with backfill_pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.observation_artifacts")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM benchmarks_v2.metric_evaluation_inputs")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM benchmarks_v2.metric_values")
        assert cur.fetchone() == (4,)
        cur.execute("SELECT count(*) FROM benchmarks_v2.metric_values_by_bucket")
        assert cur.fetchone() == (8,)
    backfill_pg.rollback()

    rerun = backfill(
        backfill_pg,
        min_result_id=result_id,
        max_result_id=result_id,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=True,
        artifact_bucket="backfill-artifacts",
    )
    assert rerun["created"] == 0
    assert rerun["reconciled"] == 1
    assert rerun["cutover_ready"]


def test_live_timing_only_failed_tts_is_read_only_and_idempotent(
    backfill_pg: psycopg.Connection[Any],
) -> None:
    _migrate(backfill_pg)
    manifest = migration._packaged_manifest("tts-v1")
    assert manifest is not None
    prompt, samples = next(iter(manifest.tts_samples.items()))
    assert len(samples) == 1
    run_id = _insert_run(
        backfill_pg,
        dataset_id="tts-v1",
        dataset_sha256=manifest.sha256,
    )
    observation_id, evaluation_id, timing_id = uuid4(), uuid4(), uuid4()
    _, timing_payload, _, _, _ = prepare_timing_events({"ttfa_ms": 12.0})
    timing_digest = hashlib.sha256(timing_payload).hexdigest()
    with backfill_pg.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.results
               (run_id,provider,model,voice,benchmark,metric_type,metric_value,metric_units,
                audio_filename,transcript,status,error,created_at)
               VALUES (%s,'provider','model','voice','TTS','TTFA',12,'milliseconds',
                       NULL,%s,'failed','provider failed',%s) RETURNING id""",
            (run_id, prompt, _NOW),
        )
        result = cur.fetchone()
        assert result is not None
        result_id = int(result[0])
        cur.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (id,run_id,dataset_id,dataset_sha256,sample_id,provider,model,voice,benchmark,
                source_kind,captured_at,status,error,failure_origin)
               VALUES (%s,%s,'tts-v1',%s,%s,'provider','model','voice','TTS',
                       'generated_audio',%s,'failed','provider failed','provider')""",
            (observation_id, run_id, manifest.sha256, samples[0], _NOW),
        )
        cur.execute(
            """INSERT INTO benchmarks_v2.observation_artifacts
               (id,observation_id,artifact_type,schema_name,schema_version,gcs_uri,
                content_sha256,size_bytes,duration_ms)
               VALUES (%s,%s,'timing_events','TimingEvents','v1',%s,%s,%s,NULL)""",
            (
                timing_id,
                observation_id,
                f"gs://private/observation-artifacts/v1/timing_events/{timing_digest[:2]}/{timing_digest}.json",
                timing_digest,
                len(timing_payload),
            ),
        )
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (id,observation_id,metric_type,metric_version,evaluation_variant,executor,status)
               VALUES (%s,%s,'TTFA','v1','default','inline','queued')""",
            (evaluation_id, observation_id),
        )
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_evaluation_inputs
               (metric_evaluation_id,observation_artifact_id,input_role,input_order)
               VALUES (%s,%s,'timing',0)""",
            (evaluation_id, timing_id),
        )
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status='running',started_at=%s WHERE id=%s""",
            (_NOW + timedelta(seconds=1), evaluation_id),
        )
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status='failed',finished_at=%s,error='provider failed' WHERE id=%s""",
            (_NOW + timedelta(seconds=2), evaluation_id),
        )
    backfill_pg.commit()

    first = backfill(
        backfill_pg,
        min_result_id=result_id,
        max_result_id=result_id,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=False,
    )
    second = backfill(
        backfill_pg,
        min_result_id=result_id,
        max_result_id=result_id,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=False,
    )
    assert first == second
    assert first["live_owned"] == 1
    assert first["reconciled"] == 1
    assert first["eligible"] == 0
    assert first["parity_mismatch_count"] == 0
    assert first["cutover_ready"]
    assert first["window"] == {
        "min_result_id": result_id,
        "max_result_id": result_id,
        "start": _WINDOW_START.isoformat(),
        "end": _WINDOW_END.isoformat(),
        "batch_size": 1,
    }
    with backfill_pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.benchmark_observations")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM benchmarks_v2.observation_artifacts")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM benchmarks_v2.metric_values")
        assert cur.fetchone() == (0,)
    backfill_pg.rollback()


def test_live_owned_incomplete_child_payload_blocks_readiness(
    backfill_pg: psycopg.Connection[Any],
) -> None:
    _migrate(backfill_pg)
    run_id = _insert_run(backfill_pg)
    result_id = _insert_wer(backfill_pg, run_id, "live.wav")
    sample_id = migration._legacy_sample("historical-stt", "live.wav")
    with backfill_pg.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (id,run_id,dataset_id,dataset_sha256,sample_id,provider,model,benchmark,
                source_kind,captured_at,status)
               VALUES (%s,%s,'historical-stt',%s,%s,'provider','model','STT',
                       'dataset_audio',%s,'succeeded')""",
            (uuid4(), run_id, _SHA, sample_id, _NOW),
        )
    backfill_pg.commit()

    report = backfill(
        backfill_pg,
        min_result_id=result_id,
        max_result_id=result_id,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=False,
    )
    assert report["live_owned"] == 1
    assert not report["cutover_ready"]
    assert report["parity_mismatches"][0]["reason"] == "evaluation"


def test_batch_one_commits_when_batch_two_fails_then_rerun_resumes(
    backfill_pg: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _migrate(backfill_pg)
    run_id = _insert_run(backfill_pg)
    first = _insert_wer(backfill_pg, run_id, "one.wav", transcript="one")
    second = _insert_wer(backfill_pg, run_id, "two.wav", transcript="two")
    fake_storage = _Storage()
    monkeypatch.setattr(gcs_storage, "Client", lambda: fake_storage)
    original = migration._insert_plan

    def fail_second(*args: Any, **kwargs: Any) -> None:
        plan = args[1]
        if (
            kwargs.get("apply", args[2] if len(args) > 2 else False)
            and plan.first.filename == "two.wav"
        ):
            raise RuntimeError("forced second batch failure")
        original(*args, **kwargs)

    monkeypatch.setattr(migration, "_insert_plan", fail_second)
    with pytest.raises(RuntimeError, match="forced second batch failure"):
        backfill(
            backfill_pg,
            min_result_id=first,
            max_result_id=second,
            window_start=_WINDOW_START,
            window_end=_WINDOW_END,
            batch_size=1,
            apply=True,
            artifact_bucket="backfill-artifacts",
        )
    with backfill_pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.benchmark_observations")
        assert cur.fetchone() == (1,)
    backfill_pg.rollback()

    monkeypatch.setattr(migration, "_insert_plan", original)
    report = backfill(
        backfill_pg,
        min_result_id=first,
        max_result_id=second,
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        batch_size=1,
        apply=True,
        artifact_bucket="backfill-artifacts",
    )
    assert report["created"] == 1
    assert report["reconciled"] == 1
    assert report["cutover_ready"]


def test_artifact_descriptor_expectation_is_content_addressed() -> None:
    payload = b'{"schema_version":"v1","transcript":"words"}'
    digest = hashlib.sha256(payload).hexdigest()
    artifact = ObservationArtifact(
        artifact_type=ObservationArtifactType.PROVIDER_TRANSCRIPT,
        schema_name="ProviderTranscript",
        schema_version="v1",
        gcs_uri=f"gs://bucket/observation-artifacts/v1/provider_transcript/{digest[:2]}/{digest}.json",
        content_sha256=digest,
        size_bytes=len(payload),
    )
    assert artifact.content_sha256 == digest
