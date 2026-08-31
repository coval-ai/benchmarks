# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Safety and real-Postgres coverage for the normalized storage backfill."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections import Counter
from contextlib import nullcontext
from dataclasses import astuple, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
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
                after = int(params[2])
                self.result = [(run_id,) for run_id in (10, 20) if run_id > after][: int(params[3])]
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
    ids, rows = migration._run_page(conn, 1, 99, 0, 1)  # type: ignore[arg-type]
    next_ids, next_rows = migration._run_page(conn, 1, 99, ids[-1], 1)  # type: ignore[arg-type]
    assert (ids, [row.run_id for row in rows]) == ([10], [10])
    assert (next_ids, [row.run_id for row in next_rows]) == ([20], [20])
    assert [call[1][2] for call in conn.value.calls if call[0] == migration._PAGE_RUN_IDS_SQL] == [
        0,
        10,
    ]


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
            cursor = params[4]
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
    assert list(migration._scheduled_buckets(conn, 1, 99, 1)) == [_NOW, later]  # type: ignore[arg-type]
    assert conn.value.calls == [
        (1, 99, 1, 99, None, None, 0, 1),
        (1, 99, 1, 99, _NOW, _NOW, 10, 1),
        (1, 99, 1, 99, later, later, 20, 1),
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
        _low: int,
        _high: int,
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


def test_runner_image_wraps_the_cli_behind_the_tolerant_entrypoint() -> None:
    dockerfile = (Path(__file__).parents[2] / "Dockerfile").read_text()
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in dockerfile
    assert 'CMD ["run"]' in dockerfile


def _invoke_entrypoint(tmp_path: Path, *args: str) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "argv.txt"
    for name in ("python", "coval-bench"):
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "{name}" "$@" > "{log}"\n')
        stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    script = Path(__file__).parents[2] / "entrypoint.sh"
    subprocess.run(  # noqa: S603 — argv is test-owned literals
        ["/bin/sh", str(script), *args], env=env, check=True
    )
    return log.read_text().splitlines()


def test_entrypoint_normalizes_both_caller_shapes(tmp_path: Path) -> None:
    short = _invoke_entrypoint(tmp_path, "run", "--kind", "stt")
    full = _invoke_entrypoint(tmp_path, "python", "-m", "coval_bench", "run", "--kind", "stt")
    assert short == ["python", "-m", "coval_bench", "run", "--kind", "stt"]
    assert full == short
    assert _invoke_entrypoint(tmp_path, "coval-bench", "db", "migrate") == [
        "coval-bench",
        "db",
        "migrate",
    ]


def test_repository_owned_container_overrides_use_the_image_entrypoint() -> None:
    root = Path(__file__).parents[2]
    overrides = {
        "docker-compose.yml": root / ".." / "docker-compose.yml",
        "runner README": root / "README.md",
        "arena snapshot": root / ".." / "scripts" / "arena-local.sh",
    }
    for name, path in overrides.items():
        assert "docker compose run --rm runner coval-bench" not in path.read_text(), name
    assert 'command: ["coval-bench"' not in overrides["docker-compose.yml"].read_text()


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
