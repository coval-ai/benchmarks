# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-Postgres coverage for the normalized readiness checker SQL."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import psycopg
from pytest_postgresql.factories import postgresql

from scripts.check_normalized_readiness import check

from .conftest import apply_migrations

readiness_pg = postgresql("pg_proc")
_SHA = "a" * 64


def _insert_observation(
    cursor: psycopg.Cursor[Any],
    *,
    run_id: int,
    sample_id: str,
    benchmark: str,
    dataset_id: str,
    source_kind: str,
    captured_at: datetime,
) -> UUID:
    cursor.execute(
        """
        INSERT INTO benchmarks_v2.benchmark_observations
          (run_id, dataset_id, dataset_sha256, sample_id, provider, model,
           benchmark, source_kind, captured_at, status)
        VALUES
          (%(run_id)s, %(dataset_id)s, %(_sha)s, %(sample_id)s, 'provider', 'model',
           %(benchmark)s, %(source_kind)s, %(captured_at)s, 'succeeded')
        RETURNING id
        """,
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "_sha": _SHA,
            "sample_id": sample_id,
            "benchmark": benchmark,
            "source_kind": source_kind,
            "captured_at": captured_at,
        },
    )
    row = cursor.fetchone()
    assert row is not None
    return UUID(str(row[0]))


def _insert_evaluation(
    cursor: psycopg.Cursor[Any],
    *,
    observation_id: UUID,
    metric_type: str,
    unit: str,
    values: Sequence[tuple[str, float, str]],
    lifecycle_at: datetime,
) -> None:
    cursor.execute(
        """
        INSERT INTO benchmarks_v2.metric_evaluations
          (observation_id, metric_type, metric_version, evaluation_variant,
           executor, status)
        VALUES (%s, %s, 'v1', 'default', 'test', 'queued')
        RETURNING id
        """,
        (observation_id, metric_type),
    )
    row = cursor.fetchone()
    assert row is not None
    evaluation_id = UUID(str(row[0]))
    cursor.execute(
        """
        UPDATE benchmarks_v2.metric_evaluations
        SET status = 'running', started_at = %(at)s, updated_at = %(at)s
        WHERE id = %(id)s
        """,
        {"id": evaluation_id, "at": lifecycle_at},
    )
    cursor.executemany(
        """
        INSERT INTO benchmarks_v2.metric_values
          (metric_evaluation_id, value_key, unit, value, value_role)
        VALUES (%s, %s, %s, %s, %s)
        """,
        [(evaluation_id, value_key, unit, value, role) for value_key, value, role in values],
    )
    cursor.execute(
        """
        UPDATE benchmarks_v2.metric_evaluations
        SET status = 'succeeded', finished_at = %(at)s, updated_at = %(at)s
        WHERE id = %(id)s
        """,
        {"id": evaluation_id, "at": lifecycle_at},
    )


def _insert_matching_rows(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT now()")
        now_row = cursor.fetchone()
        assert now_row is not None
        now = now_row[0]
        old_at = now - timedelta(hours=4)
        legacy_at = now - timedelta(minutes=30)
        normalized_at = now - timedelta(minutes=90)

        cursor.execute(
            """
            INSERT INTO benchmarks_v2.runs
              (started_at, finished_at, runner_sha, dataset_id, dataset_sha256,
               status, scheduled_at)
            VALUES
              (%(old_at)s, %(old_at)s, 'runner-sha', 'stt-v1', %(_sha)s,
               'succeeded', %(old_at)s),
              (%(legacy_at)s, %(legacy_at)s, 'runner-sha', 'stt-v1', %(_sha)s,
               'succeeded', %(legacy_at)s),
              (%(legacy_at)s, %(legacy_at)s, 'runner-sha', 'source-tts', %(_sha)s,
               'succeeded', %(legacy_at)s)
            RETURNING id
            """,
            {"old_at": old_at, "legacy_at": legacy_at, "_sha": _SHA},
        )
        run_rows = cursor.fetchall()
        assert len(run_rows) == 3
        old_run_id, stt_run_id, tts_run_id = (int(row[0]) for row in run_rows)

        cursor.executemany(
            """
            INSERT INTO benchmarks_v2.results
              (run_id, provider, model, benchmark, metric_type, metric_value,
               metric_units, status, created_at, wer_insertions_pct,
               wer_deletions_pct, wer_substitutions_pct)
            VALUES
              (%s, 'provider', 'model', %s, %s, %s, %s, 'success', %s, %s, %s, %s)
            """,
            [
                (old_run_id, "STT", "WER", 6.0, "percent", old_at, 1.0, 2.0, 3.0),
                (stt_run_id, "STT", "WER", 6.0, "percent", legacy_at, 1.0, 2.0, 3.0),
                (stt_run_id, "STT", "WER", 6.0, "percent", legacy_at, 1.0, 2.0, 3.0),
                (tts_run_id, "TTS", "TTFA", 12.0, "milliseconds", legacy_at, None, None, None),
                (
                    tts_run_id,
                    "TTS",
                    "TTFARoundtrip",
                    10.0,
                    "milliseconds",
                    legacy_at,
                    None,
                    None,
                    None,
                ),
                (
                    tts_run_id,
                    "TTS",
                    "TTFALeadingSilence",
                    2.0,
                    "milliseconds",
                    legacy_at,
                    None,
                    None,
                    None,
                ),
            ],
        )

        for sample_id in ("stt-sample-1", "stt-sample-2"):
            observation_id = _insert_observation(
                cursor,
                run_id=stt_run_id,
                sample_id=sample_id,
                benchmark="STT",
                dataset_id="stt-v1",
                source_kind="dataset_audio",
                captured_at=normalized_at,
            )
            _insert_evaluation(
                cursor,
                observation_id=observation_id,
                metric_type="WER",
                unit="percent",
                values=[
                    ("primary", 6.0, "primary"),
                    ("insertions", 1.0, "component"),
                    ("deletions", 2.0, "component"),
                    ("substitutions", 3.0, "component"),
                ],
                lifecycle_at=now,
            )

        tts_observation_id = _insert_observation(
            cursor,
            run_id=tts_run_id,
            sample_id="tts-sample",
            benchmark="TTS",
            dataset_id="tts-v1",
            source_kind="generated_audio",
            captured_at=normalized_at,
        )
        _insert_evaluation(
            cursor,
            observation_id=tts_observation_id,
            metric_type="TTFA",
            unit="milliseconds",
            values=[
                ("primary", 12.0, "primary"),
                ("roundtrip", 10.0, "component"),
                ("leading_silence", 2.0, "component"),
            ],
            lifecycle_at=now,
        )

        bucket_at = legacy_at.replace(minute=0, second=0, microsecond=0)
        cursor.execute(
            """
            INSERT INTO benchmarks_v2.results_by_bucket
              (provider, model, benchmark, dataset_id, metric_type, bucket_at,
               min_value, p25, p50, p75, max_value, value_sum, sample_count)
            VALUES
              ('provider', 'model', 'STT', 'stt-v1', 'WER', %(bucket_at)s,
               6, 6, 6, 6, 6, 12, 2)
            """,
            {"bucket_at": bucket_at},
        )
        cursor.execute(
            """
            INSERT INTO benchmarks_v2.metric_values_by_bucket
              (provider, model, benchmark, dataset_id, metric_type,
               metric_version, evaluation_variant, value_key, unit, bucket_at,
               min_value, p25, p50, p75, max_value, value_sum, sample_count)
            VALUES
              ('provider', 'model', 'STT', 'stt-v1', 'WER',
               'v1', 'default', 'primary', 'percent', %(bucket_at)s,
               6, 6, 6, 6, 6, 12, 2)
            """,
            {"bucket_at": bucket_at},
        )
    conn.commit()


def test_real_sql_matches_wer_duplicates_ttfa_components_and_rollups(
    readiness_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(readiness_pg)
    _insert_matching_rows(readiness_pg)

    report = check(
        readiness_pg,
        required_hours=2,
        chunk_hours=1,
        statement_timeout_seconds=5,
        total_timeout_seconds=30,
    )
    readiness_pg.rollback()

    assert report["ready"] is True
    assert report["coverage_hours"] >= 2
    assert report["raw_mismatch_count"] == 0
    assert report["rollup_mismatch_count"] == 0
