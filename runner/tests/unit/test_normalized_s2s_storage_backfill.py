# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Safety and real-Postgres coverage for the normalized S2S backfill."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import uuid
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from click.testing import CliRunner
from pytest_postgresql.factories import postgresql

from coval_bench.migrations import backfill_normalized_s2s_storage as migration
from coval_bench.migrations.backfill_normalized_s2s_storage import (
    LegacyS2SRow,
    ProgressReporter,
    _plans,
    backfill,
    backfill_normalized_s2s_storage_cli,
)
from scripts import benchmark_normalized_s2s_backfill as benchmark

from .conftest import apply_migrations

s2s_pg = postgresql("pg_proc")
_NOW = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
_SHA = "a" * 64
_UNNORMALIZED_PROVENANCE = "persona/test-set-v1"


def _dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    auth = f"{info.user}:{info.password}@" if info.password else f"{info.user}@"
    return (
        f"postgresql://{auth}{info.host or 'localhost'}:{info.port or 5432}/{info.dbname or 'test'}"
    )


def _truncate(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE benchmarks_v2.runs, benchmarks_v2.metric_values_by_bucket "
            "RESTART IDENTITY CASCADE"
        )
    conn.commit()


@pytest.fixture
def s2s_db(request: pytest.FixtureRequest) -> Any:
    external_url = os.environ.get("S2S_BACKFILL_TEST_DATABASE_URL")
    if external_url:
        conn = psycopg.connect(external_url)
        _truncate(conn)
        try:
            yield conn
        finally:
            _truncate(conn)
            conn.close()
        return
    conn = request.getfixturevalue("s2s_pg")
    apply_migrations(conn)
    yield conn


def _row(
    metric: str,
    *,
    value: float | None = 1.0,
    status: str = "success",
    error: str | None = None,
    row_id: int = 1,
    run_id: int = 4,
    sample_id: str | None = "coval-run/simulation",
    dataset_id: str = "s2s-v1",
    dataset_sha256: str = _UNNORMALIZED_PROVENANCE,
    scheduled_at: datetime | None = _NOW,
    captured_at: datetime = _NOW,
) -> LegacyS2SRow:
    return LegacyS2SRow(
        row_id,
        run_id,
        dataset_id,
        dataset_sha256,
        scheduled_at,
        "provider",
        "model",
        "voice",
        metric,
        value,
        {
            "V2V": "milliseconds",
            "InstructionFollowing": "percent",
            "InterruptionRate": "per_minute",
        }[metric],
        sample_id,
        status,
        error,
        captured_at,
    )


def _insert_run(
    conn: psycopg.Connection[Any],
    *,
    status: str = "succeeded",
    scheduled_at: datetime | None = _NOW,
    dataset_id: str = "persona-v1",
    dataset_sha256: str = _UNNORMALIZED_PROVENANCE,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.runs
               (started_at,finished_at,runner_sha,dataset_id,dataset_sha256,status,scheduled_at)
               VALUES (%s,%s,'runner-sha',%s,%s,%s,%s) RETURNING id""",
            (
                _NOW,
                None if status == "running" else _NOW + timedelta(minutes=1),
                dataset_id,
                dataset_sha256,
                status,
                scheduled_at,
            ),
        )
        row = cur.fetchone()
        assert row is not None
    conn.commit()
    return int(row[0])


def _insert_result(
    conn: psycopg.Connection[Any],
    run_id: int,
    metric: str,
    *,
    value: float | None = 1.0,
    status: str = "success",
    error: str | None = None,
    sample_id: str = "coval-run/simulation",
    provider: str = "provider",
    model: str = "model",
    voice: str | None = "voice",
) -> int:
    unit = {
        "V2V": "milliseconds",
        "InstructionFollowing": "percent",
        "InterruptionRate": "per_minute",
    }[metric]
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.results
               (run_id,provider,model,voice,benchmark,metric_type,metric_value,metric_units,
                audio_filename,status,error,created_at)
               VALUES (%s,%s,%s,%s,'S2S',%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (
                run_id,
                provider,
                model,
                voice,
                metric,
                value,
                unit,
                sample_id,
                status,
                error,
                _NOW + timedelta(seconds=30),
            ),
        )
        row = cur.fetchone()
        assert row is not None
    conn.commit()
    return int(row[0])


def _seed_conversation(
    conn: psycopg.Connection[Any],
    *,
    run_status: str = "succeeded",
    scheduled_at: datetime | None = _NOW,
    sample_id: str = "coval-run/simulation",
    metrics: tuple[tuple[str, float | None, str], ...] = (
        ("V2V", 210.0, "success"),
        ("InstructionFollowing", 91.0, "success"),
        ("InterruptionRate", None, "failed"),
    ),
) -> tuple[int, int, int]:
    run_id = _insert_run(conn, status=run_status, scheduled_at=scheduled_at)
    ids = [
        _insert_result(
            conn,
            run_id,
            metric,
            value=value,
            status=status,
            sample_id=sample_id,
        )
        for metric, value, status in metrics
    ]
    return run_id, min(ids), max(ids)


def _events(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_groups_mixed_metrics_and_preserves_exact_recoverable_payload() -> None:
    skipped: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    plans = _plans(
        [
            _row("V2V"),
            _row("InstructionFollowing"),
            _row("InterruptionRate", value=None, status="failed", error=" exact error "),
        ],
        skipped,
        conflicts,
    )

    assert len(plans) == 1
    assert plans[0].status == "succeeded"
    assert plans[0].error is None
    assert plans[0].dataset_sha256 == hashlib.sha256(_UNNORMALIZED_PROVENANCE.encode()).hexdigest()
    assert plans[0].evaluations()[-1] == (
        "V2V",
        "succeeded",
        1.0,
        None,
    )
    assert (
        next(item for item in plans[0].evaluations() if item[0] == "InterruptionRate")[3]
        == " exact error "
    )
    assert skipped == {}
    assert conflicts == {}


def test_all_failed_uses_canonical_fallbacks() -> None:
    plans = _plans([_row("V2V", value=None, status="failed")], Counter(), Counter())
    assert plans[0].status == "failed"
    assert plans[0].error == "Coval conversation produced no successful metric value"
    assert plans[0].evaluations()[0][3] == "legacy metric produced no value"


def test_duplicate_metrics_and_conflicting_groups_fail_closed() -> None:
    skipped: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    assert _plans([_row("V2V"), _row("V2V", row_id=2)], skipped, conflicts) == []
    assert conflicts == {"conflicting_group": 1}

    conflicts.clear()
    rows = [_row("V2V"), replace(_row("InstructionFollowing", row_id=2), dataset_id="other")]
    assert _plans(rows, skipped, conflicts) == []
    assert conflicts == {"conflicting_group": 1}


def test_progress_cadence_throughput_eta_and_durable_checkpoint() -> None:
    clock = _Clock()
    stream = io.StringIO()
    report = migration._report(1, 99, 1)
    reporter = ProgressReporter(
        "apply", 1, 99, 1, stream=stream, monotonic=clock, interval_seconds=30, max_units=10
    )
    reporter.total_runs = 4
    reporter.phase_started("source_reconciliation", report, total_runs=4)
    clock.value = 31
    row = _row("V2V", row_id=9, run_id=7, sample_id="private/sample")
    reporter.completed_page([7], [row], 1, report, phase="source_reconciliation", committed=True)

    event = _events(stream)[-1]
    assert event["status"] == "progress"
    assert event["last_committed_run_id"] == 7
    assert event["last_committed_result_id"] == 9
    assert event["throughput_results_per_second"] == pytest.approx(1 / 31, rel=1e-5)
    assert event["throughput_conversations_per_second"] == pytest.approx(1 / 31, rel=1e-5)
    assert event["eta_seconds"] == pytest.approx(93)
    assert "private/sample" not in stream.getvalue()


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://postgres@127.0.0.1/postgres?hostaddr=203.0.113.1",
        "postgresql://postgres@127.0.0.1/postgres?dbname=production",
        "postgresql://postgres@127.0.0.1/postgres?service=remote",
        "postgresql://postgres@127.0.0.1/postgres#dbname=production",
    ],
)
def test_benchmark_rejects_libpq_routing_and_database_overrides(url: str) -> None:
    with pytest.raises(SystemExit):
        benchmark._validate_local_url(argparse.ArgumentParser(), url)

    benchmark._validate_local_url(
        argparse.ArgumentParser(), "postgresql://postgres@127.0.0.1/postgres"
    )


def test_apply_reconciles_payload_lineage_rollups_and_is_idempotent(
    s2s_db: psycopg.Connection[Any],
) -> None:
    successful_metrics = (
        ("V2V", 210.0, "success"),
        ("InstructionFollowing", 91.0, "success"),
        ("InterruptionRate", 0.2, "success"),
    )
    _, low, _ = _seed_conversation(
        s2s_db,
        sample_id="coval-run-one/simulation",
        metrics=successful_metrics,
    )
    _, _, high = _seed_conversation(
        s2s_db,
        sample_id="coval-run-two/simulation",
        metrics=successful_metrics,
    )
    stream = io.StringIO()
    first = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        batch_size=1,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 1, stream=stream),
    )

    assert first["backfill_complete"] is True
    assert (first["created"], first["evaluations"], first["values"]) == (2, 6, 6)
    assert first["verification_reconciled"] == 2
    assert first["public_parity_mismatch_count"] == 0
    assert first["rollup_mismatch_count"] == 0
    with s2s_db.cursor() as cur:
        cur.execute("SELECT dataset_sha256 FROM benchmarks_v2.benchmark_observations")
        assert cur.fetchone() == (hashlib.sha256(_UNNORMALIZED_PROVENANCE.encode()).hexdigest(),)
        cur.execute(
            """SELECT
                 (SELECT count(*) FROM benchmarks_v2.observation_artifacts),
                 (SELECT count(*) FROM benchmarks_v2.preprocessing_artifacts),
                 (SELECT count(*) FROM benchmarks_v2.metric_evaluation_inputs),
                 (SELECT count(*) FROM benchmarks_v2.metric_artifacts)"""
        )
        assert cur.fetchone() == (0, 0, 0, 0)
        cur.execute("SELECT count(*) FROM benchmarks_v2.metric_values_by_bucket")
        assert cur.fetchone() == (6,)
    s2s_db.commit()

    second = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        batch_size=1,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 1, stream=io.StringIO()),
    )
    assert second["backfill_complete"] is True
    assert second["created"] == 0
    assert second["reconciled"] == 2
    phases = {(event["phase"], event["status"]) for event in _events(stream)}
    for phase in (
        "operation",
        "qualifying_run_count",
        "source_reconciliation",
        "post_write_verification",
        "public_parity",
        "rollup_verification",
    ):
        assert (phase, "started") in phases
        assert (phase, "completed") in phases


def test_public_parity_is_bounded_to_complete_run_pages(
    s2s_db: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ids: list[int] = []
    low = 0
    high = 0
    for index in range(3):
        run_id, first, last = _seed_conversation(
            s2s_db,
            sample_id=f"coval-run-{index}/simulation",
            metrics=(("V2V", 210.0 + index, "success"),),
        )
        run_ids.append(run_id)
        low = first if low == 0 else min(low, first)
        high = max(high, last)
    initial = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        batch_size=1,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 1, stream=io.StringIO()),
    )
    assert initial["backfill_complete"] is True

    original_verify = migration._verify_public_parity_page
    verified_pages: list[list[int]] = []

    def record_page(
        cur: psycopg.Cursor[Any],
        page_run_ids: list[int],
        page_low: int,
        page_high: int,
        report: dict[str, Any],
    ) -> None:
        verified_pages.append(list(page_run_ids))
        original_verify(cur, page_run_ids, page_low, page_high, report)

    monkeypatch.setattr(migration, "_verify_public_parity_page", record_page)
    _insert_live_owned(s2s_db, run_ids[-1], sample_id="normalized-only/sample")
    report = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        batch_size=1,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 1, stream=io.StringIO()),
    )

    assert verified_pages == [[run_id] for run_id in run_ids]
    assert report["public_parity_mismatch_count"] == 1
    assert report["public_parity_mismatches"][0]["legacy_count"] == 0
    assert report["public_parity_mismatches"][0]["normalized_count"] == 1
    assert report["backfill_complete"] is False


def test_public_parity_sets_transaction_local_timeout_before_page_query() -> None:
    class RecordingCursor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def execute(self, query: str, params: object = None) -> None:
            self.calls.append((query, params))

        def fetchall(self) -> list[tuple[Any, ...]]:
            return []

    cursor = RecordingCursor()
    report = migration._report(11, 29, 1)
    migration._verify_public_parity_page(cursor, [17], 11, 29, report)  # type: ignore[arg-type]

    assert cursor.calls == [
        (
            "SELECT set_config('statement_timeout', %s, true)",
            (migration._PUBLIC_PARITY_STATEMENT_TIMEOUT,),
        ),
        (
            migration._PUBLIC_PARITY_PAGE_SQL,
            {"run_ids": [17], "low": 11, "high": 29},
        ),
    ]
    assert migration._PUBLIC_PARITY_STATEMENT_TIMEOUT == "30s"


def test_entirely_failed_conversation_persists_canonical_errors(
    s2s_db: psycopg.Connection[Any],
) -> None:
    _, low, high = _seed_conversation(
        s2s_db,
        sample_id="coval-run/entirely-failed",
        metrics=(("V2V", None, "failed"),),
    )
    report = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 100, stream=io.StringIO()),
    )
    assert report["backfill_complete"] is True
    with s2s_db.cursor() as cur:
        cur.execute(
            """SELECT o.error,e.error
               FROM benchmarks_v2.benchmark_observations o
               JOIN benchmarks_v2.metric_evaluations e ON e.observation_id=o.id"""
        )
        assert cur.fetchone() == (
            "Coval conversation produced no successful metric value",
            "legacy metric produced no value",
        )
    s2s_db.commit()


def test_extra_failed_normalized_observation_blocks_completion(
    s2s_db: psycopg.Connection[Any],
) -> None:
    run_id, low, high = _seed_conversation(s2s_db)
    backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 100, stream=io.StringIO()),
    )
    with s2s_db.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (id,run_id,dataset_id,dataset_sha256,sample_id,provider,model,voice,
                benchmark,source_kind,captured_at,status,error,failure_origin)
               VALUES (%s,%s,'persona-v1',%s,'unexpected/sample','provider','model','voice',
                       'S2S','conversation_audio',%s,'failed','extra normalized row','provider')""",
            (uuid.uuid4(), run_id, _SHA, _NOW),
        )
    s2s_db.commit()

    report = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 100, stream=io.StringIO()),
    )
    assert report["backfill_complete"] is False
    assert report["observation_population_mismatch_count"] == 1
    assert report["observation_population_mismatches"][0]["expected_count"] == 0


def test_rollup_tamper_is_detected_exactly_by_read_only_dry_run(
    s2s_db: psycopg.Connection[Any],
) -> None:
    _, low, high = _seed_conversation(s2s_db, metrics=(("V2V", 210.0, "success"),))
    backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 100, stream=io.StringIO()),
    )
    with s2s_db.cursor() as cur:
        cur.execute("UPDATE benchmarks_v2.metric_values_by_bucket SET value_sum=value_sum+0.000001")
    s2s_db.commit()

    report = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=False,
        reporter=ProgressReporter("dry_run", low, high, 100, stream=io.StringIO()),
    )
    assert report["eligible"] == 0
    assert report["backfill_complete"] is False
    assert report["rollup_mismatch_count"] == 4


def test_s2s_refresh_preserves_and_ignores_same_bucket_non_s2s_rollups(
    s2s_db: psycopg.Connection[Any],
) -> None:
    _, low, high = _seed_conversation(s2s_db, metrics=(("V2V", 210.0, "success"),))
    with s2s_db.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
                (provider,model,benchmark,dataset_id,metric_type,metric_version,
                 evaluation_variant,value_key,unit,bucket_at,min_value,p25,p50,p75,
                 max_value,value_sum,sample_count)
                VALUES
                ('stt-provider','stt-model','STT','stt-dataset','WER','v7','custom',
                 'primary','percent',%s,11,12,13,14,15,123,7),
                ('tts-provider','tts-model','TTS','__all__','MOS','v9','custom',
                 'primary','score',%s,21,22,23,24,25,456,8)""",
            (_NOW, _NOW),
        )
        cur.execute(
            """SELECT provider,model,benchmark,dataset_id,metric_type,metric_version,
                       evaluation_variant,value_key,unit,bucket_at,min_value,p25,p50,p75,
                       max_value,value_sum,sample_count
                FROM benchmarks_v2.metric_values_by_bucket
                WHERE benchmark <> 'S2S' ORDER BY benchmark"""
        )
        before = cur.fetchall()
    s2s_db.commit()

    applied = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 100, stream=io.StringIO()),
    )
    assert applied["backfill_complete"] is True
    assert applied["rollup_mismatch_count"] == 0
    with s2s_db.cursor() as cur:
        cur.execute(
            """SELECT provider,model,benchmark,dataset_id,metric_type,metric_version,
                       evaluation_variant,value_key,unit,bucket_at,min_value,p25,p50,p75,
                       max_value,value_sum,sample_count
                FROM benchmarks_v2.metric_values_by_bucket
                WHERE benchmark <> 'S2S' ORDER BY benchmark"""
        )
        assert cur.fetchall() == before
        cur.execute(
            "UPDATE benchmarks_v2.metric_values_by_bucket "
            "SET value_sum=value_sum+1 WHERE benchmark='STT'"
        )
    s2s_db.commit()

    non_s2s_tamper = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=False,
        reporter=ProgressReporter("dry_run", low, high, 100, stream=io.StringIO()),
    )
    assert non_s2s_tamper["backfill_complete"] is True
    assert non_s2s_tamper["rollup_mismatch_count"] == 0

    with s2s_db.cursor() as cur:
        cur.execute(
            "UPDATE benchmarks_v2.metric_values_by_bucket "
            "SET value_sum=value_sum+0.000001 WHERE benchmark='S2S'"
        )
    s2s_db.commit()
    s2s_tamper = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=False,
        reporter=ProgressReporter("dry_run", low, high, 100, stream=io.StringIO()),
    )
    assert s2s_tamper["backfill_complete"] is False
    assert s2s_tamper["rollup_mismatch_count"] == 4


def _insert_live_owned(
    conn: psycopg.Connection[Any],
    run_id: int,
    *,
    dataset_id: str = "persona-v1",
    sample_id: str = "coval-run/simulation",
) -> uuid.UUID:
    observation_id = uuid.uuid4()
    evaluation_id = uuid.uuid4()
    normalized_sha = hashlib.sha256(_UNNORMALIZED_PROVENANCE.encode()).hexdigest()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (id,run_id,dataset_id,dataset_sha256,sample_id,provider,model,voice,
                benchmark,source_kind,captured_at,status)
               VALUES (%s,%s,%s,%s,%s,'provider','model','voice',
                       'S2S','conversation_audio',%s,'succeeded')""",
            (
                observation_id,
                run_id,
                dataset_id,
                normalized_sha,
                sample_id,
                _NOW + timedelta(seconds=30),
            ),
        )
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (id,observation_id,metric_type,metric_version,evaluation_variant,executor,status)
               VALUES (%s,%s,'V2V','v1','default','coval_api','queued')""",
            (evaluation_id, observation_id),
        )
        cur.execute(
            "UPDATE benchmarks_v2.metric_evaluations "
            "SET status='running',started_at=%s WHERE id=%s",
            (_NOW + timedelta(seconds=31), evaluation_id),
        )
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_values
               (metric_evaluation_id,value_key,unit,value,value_role)
               VALUES (%s,'primary','milliseconds',210,'primary')""",
            (evaluation_id,),
        )
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status='succeeded',finished_at=%s WHERE id=%s""",
            (_NOW + timedelta(seconds=32), evaluation_id),
        )
    conn.commit()
    return observation_id


def test_exact_live_owned_random_ids_reconcile(
    s2s_db: psycopg.Connection[Any],
) -> None:
    run_id, low, high = _seed_conversation(s2s_db, metrics=(("V2V", 210.0, "success"),))
    observation_id = _insert_live_owned(s2s_db, run_id)

    report = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 100, stream=io.StringIO()),
    )
    assert report["backfill_complete"] is True
    assert report["created"] == 0
    assert report["live_owned"] == 1
    assert report["verification_live_owned"] == 1
    with s2s_db.cursor() as cur:
        cur.execute("SELECT id FROM benchmarks_v2.benchmark_observations")
        assert cur.fetchone() == (observation_id,)
    s2s_db.commit()


def test_conflicting_live_owned_payload_blocks_completion(
    s2s_db: psycopg.Connection[Any],
) -> None:
    run_id, low, high = _seed_conversation(s2s_db, metrics=(("V2V", 210.0, "success"),))
    _insert_live_owned(s2s_db, run_id, dataset_id="wrong-persona")

    report = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 100, stream=io.StringIO()),
    )
    assert report["backfill_complete"] is False
    assert report["conflicts_by_reason"] == {"observation_payload_mismatch": 1}
    assert report["verification_conflicts_by_reason"] == {"observation_payload_mismatch": 1}
    assert report["payload_mismatch_count"] == 1


def test_split_window_nonterminal_and_dry_run_no_dml(
    s2s_db: psycopg.Connection[Any],
) -> None:
    split_run = _insert_run(s2s_db)
    first = _insert_result(s2s_db, split_run, "V2V", value=10)
    running_run = _insert_run(s2s_db, status="running")
    last_in_window = _insert_result(s2s_db, running_run, "V2V", value=11)
    _insert_result(s2s_db, split_run, "InstructionFollowing", value=90)

    with s2s_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.benchmark_observations")
        before = cur.fetchone()
    s2s_db.commit()
    report = backfill(
        s2s_db,
        min_result_id=first,
        max_result_id=last_in_window,
        batch_size=1,
        apply=False,
        reporter=ProgressReporter("dry_run", first, last_in_window, 1, stream=io.StringIO()),
    )
    with s2s_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.benchmark_observations")
        after = cur.fetchone()
    s2s_db.commit()

    assert before == after == (0,)
    assert report["backfill_complete"] is False
    assert report["skipped_by_reason"]["split_window_run"] == 1
    assert report["skipped_by_reason"]["run_not_terminal"] == 1
    assert report["verification_skipped_by_reason"]["split_window_run"] == 1
    assert report["verification_skipped_by_reason"]["run_not_terminal"] == 1


def test_page_failure_rolls_back_only_current_page_and_rerun_resumes(
    s2s_db: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    first_run, low, _ = _seed_conversation(
        s2s_db,
        scheduled_at=_NOW,
        sample_id="run-one/sample",
        metrics=(("V2V", 10.0, "success"),),
    )
    second_run, _, high = _seed_conversation(
        s2s_db,
        scheduled_at=_NOW + timedelta(days=1),
        sample_id="run-two/sample",
        metrics=(("V2V", 20.0, "success"),),
    )
    original_refresh = migration._refresh
    stream = io.StringIO()

    def fail_second(cur: psycopg.Cursor[Any], bucket: datetime) -> None:
        if bucket == _NOW + timedelta(days=1):
            raise RuntimeError("forced second-page failure")
        original_refresh(cur, bucket)

    monkeypatch.setattr(migration, "_refresh", fail_second)
    with pytest.raises(RuntimeError, match="second-page"):
        backfill(
            s2s_db,
            min_result_id=low,
            max_result_id=high,
            batch_size=1,
            apply=True,
            reporter=ProgressReporter("apply", low, high, 1, stream=stream),
        )
    with s2s_db.cursor() as cur:
        cur.execute(
            "SELECT run_id,count(*) FROM benchmarks_v2.benchmark_observations GROUP BY run_id"
        )
        assert cur.fetchall() == [(first_run, 1)]
    s2s_db.commit()
    failed = _events(stream)[-1]
    assert failed["status"] == "failed"
    assert failed["last_committed_run_id"] == first_run

    monkeypatch.setattr(migration, "_refresh", original_refresh)
    resumed = backfill(
        s2s_db,
        min_result_id=low,
        max_result_id=high,
        batch_size=1,
        apply=True,
        reporter=ProgressReporter("apply", low, high, 1, stream=io.StringIO()),
    )
    assert resumed["backfill_complete"] is True
    assert resumed["created"] == 1
    assert resumed["reconciled"] == 1
    with s2s_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.benchmark_observations")
        assert cur.fetchone() == (2,)
    s2s_db.commit()
    assert second_run != first_run


def test_cancellation_emits_checkpoint_and_releases_global_lock(
    s2s_db: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, low, high = _seed_conversation(s2s_db, metrics=(("V2V", 10.0, "success"),))
    stream = io.StringIO()
    monkeypatch.setattr(
        migration,
        "_verify_public_parity_page",
        lambda *_: (_ for _ in ()).throw(migration._Cancelled()),
    )
    with pytest.raises(migration._Cancelled):
        backfill(
            s2s_db,
            min_result_id=low,
            max_result_id=high,
            apply=True,
            reporter=ProgressReporter("apply", low, high, 1, stream=stream),
        )
    events = _events(stream)
    assert events[-2]["phase"] == "public_parity"
    assert events[-2]["status"] == "cancelled"
    assert events[-1]["phase"] == "operation"
    assert events[-1]["last_committed_run_id"] == run_id
    with s2s_db.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0))", (migration._LOCK,))
        assert cur.fetchone() == (True,)
        cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (migration._LOCK,))
    s2s_db.commit()


def test_cli_requires_frozen_apply_and_keeps_final_report_on_stdout(
    s2s_db: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, low, high = _seed_conversation(s2s_db, metrics=(("V2V", 10.0, "success"),))

    class Settings:
        database_url = _dsn(s2s_db)

    monkeypatch.setattr(migration, "get_settings", lambda: Settings())
    missing_max = CliRunner().invoke(backfill_normalized_s2s_storage_cli, ["--apply"])
    assert missing_max.exit_code == 2
    assert "--apply requires an explicit --max-result-id" in missing_max.output

    result = CliRunner().invoke(
        backfill_normalized_s2s_storage_cli,
        ["--min-result-id", str(low), "--max-result-id", str(high)],
    )
    assert result.exit_code == 0
    assert "normalized_s2s_storage_backfill_progress" not in result.stdout
    assert "normalized_s2s_storage_backfill_progress" in result.stderr
    final = json.loads(result.stdout.splitlines()[-1])
    assert final["window"]["max_result_id"] == high
    assert final["backfill_complete"] is False

    auto_frozen = CliRunner().invoke(
        backfill_normalized_s2s_storage_cli,
        ["--min-result-id", str(low)],
    )
    assert auto_frozen.exit_code == 0
    auto_final = json.loads(auto_frozen.stdout.splitlines()[-1])
    assert auto_final["window"]["max_result_id"] == high
    assert auto_final["backfill_complete"] is False
