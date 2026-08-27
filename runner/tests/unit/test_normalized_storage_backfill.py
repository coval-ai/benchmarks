# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Safety and real-Postgres coverage for the normalized storage backfill."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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

backfill_pg = postgresql("pg_proc")
_INI_PATH = Path(__file__).parents[2] / "alembic.ini"
_NOW = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
_SHA = "a" * 64


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


def _insert_run(conn: psycopg.Connection[Any]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.runs
               (started_at,finished_at,runner_sha,dataset_id,dataset_sha256,status,scheduled_at)
               VALUES (%s,%s,'runner-sha','historical-stt',%s,'succeeded',%s)
               RETURNING id""",
            (_NOW, _NOW, _SHA, _NOW),
        )
        row = cur.fetchone()
        assert row is not None
    conn.commit()
    return int(row[0])


def _insert_wer(
    conn: psycopg.Connection[Any], run_id: int, filename: str, *, transcript: str = "words"
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.results
               (run_id,provider,model,benchmark,metric_type,metric_value,metric_units,
                audio_filename,transcript,status,error,created_at,wer_insertions_pct,
                wer_deletions_pct,wer_substitutions_pct)
               VALUES (%s,'provider','model','STT','WER',6,'percent',%s,%s,'success',
                       NULL,%s,1,2,3) RETURNING id""",
            (run_id, filename, transcript, _NOW),
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


def test_tts_anchor_without_filename_is_explicitly_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_tts_sample", lambda *_: "sample")
    skipped: Counter[str] = Counter()
    plans = _tts_plans([_row("TTFA", 12.0, filename=None)], skipped)
    assert plans == []
    assert skipped["tts_anchor_filename_missing"] == 1


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


def test_cli_apply_requires_explicit_frozen_maximum() -> None:
    result = CliRunner().invoke(backfill_normalized_storage_cli, ["--apply"])
    assert result.exit_code == 2
    assert "--apply requires an explicit --max-result-id" in result.output


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
        batch_size=1,
        apply=True,
        artifact_bucket="backfill-artifacts",
    )
    assert rerun["created"] == 0
    assert rerun["reconciled"] == 1
    assert rerun["cutover_ready"]


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
        batch_size=1,
        apply=False,
    )
    assert report["live_owned"] == 1
    assert not report["cutover_ready"]
    assert report["parity_mismatches"][0]["reason"] == "live_owned_payload_mismatch"


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
