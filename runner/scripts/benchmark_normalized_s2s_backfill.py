#!/usr/bin/env python3
"""Benchmark the read-only S2S backfill on a disposable local database."""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import platform
import resource
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Self, cast
from urllib.parse import urlsplit, urlunsplit

import psycopg
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg import sql
from psycopg.abc import Params, QueryNoTemplate

from coval_bench.migrations.backfill_normalized_s2s_storage import (
    ProgressReporter,
    backfill,
)

_BATCH_SIZES = (25, 100, 400)
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_INI_PATH = Path(__file__).parents[1] / "alembic.ini"
_BASE_TIME = datetime(2026, 8, 27, tzinfo=UTC)


@dataclass
class _QueryTimer:
    seconds: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.seconds = 0.0
        self.count = 0


_QUERY_TIMER = _QueryTimer()


class _TimingCursor(psycopg.Cursor[Any]):
    """Measure client-observed execution duration for every SQL statement."""

    def execute(  # type: ignore[override]  # Template queries are not used by this harness.
        self,
        query: QueryNoTemplate,
        params: Params | None = None,
        *,
        prepare: bool | None = None,
        binary: bool | None = None,
    ) -> Self:
        started = time.monotonic()
        try:
            return super().execute(query, params, prepare=prepare, binary=binary)
        finally:
            _QUERY_TIMER.seconds += time.monotonic() - started
            _QUERY_TIMER.count += 1


def _database_url(admin_url: str, database_name: str) -> str:
    parts = urlsplit(admin_url)
    return urlunsplit(parts._replace(path=f"/{database_name}"))


def _validate_local_url(parser: argparse.ArgumentParser, url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"postgresql", "postgres"}:
        parser.error("--admin-database-url must be a PostgreSQL URL")
    if parsed.hostname not in _LOOPBACK_HOSTS:
        parser.error("benchmark refuses non-loopback database hosts")
    if not parsed.path or parsed.path == "/":
        parser.error("--admin-database-url must name an administrative database")
    if parsed.query or parsed.fragment:
        parser.error(
            "--admin-database-url must not contain query parameters or a fragment; "
            "libpq routing overrides are unsafe for a disposable benchmark"
        )


def _create_database(admin_url: str, database_name: str) -> None:
    with psycopg.connect(admin_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))


def _drop_database(admin_url: str, database_name: str) -> None:
    if not database_name.startswith("coval_s2s_backfill_bench_"):
        raise RuntimeError("refusing to drop an unexpected database name")
    with psycopg.connect(admin_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database_name)))


def _migrate(database_url: str) -> None:
    config = AlembicConfig(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    )
    alembic_command.upgrade(config, "head")


def _seed(database_url: str, *, runs: int, conversations_per_run: int) -> dict[str, int]:
    min_result_id: int | None = None
    max_result_id: int | None = None
    result_count = 0
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        for run_index in range(runs):
            captured_at = _BASE_TIME + timedelta(minutes=run_index)
            scheduled_at = _BASE_TIME + timedelta(hours=run_index)
            cur.execute(
                """INSERT INTO benchmarks_v2.runs
                   (started_at,finished_at,runner_sha,dataset_id,dataset_sha256,status,scheduled_at)
                   VALUES (%s,%s,'benchmark-runner',%s,%s,%s,%s) RETURNING id""",
                (
                    captured_at,
                    captured_at + timedelta(minutes=1),
                    f"persona-{run_index % 12}",
                    f"persona-source-{run_index % 12}",
                    "partial" if run_index % 11 == 0 else "succeeded",
                    scheduled_at,
                ),
            )
            run_row = cur.fetchone()
            if run_row is None:
                raise RuntimeError("seed run insert returned no row")
            run_id = int(run_row[0])
            values: list[tuple[Any, ...]] = []
            for conversation_index in range(conversations_per_run):
                sample_id = f"coval-run-{run_index}/simulation-{conversation_index}"
                provider = ("openai", "xai", "hume")[run_index % 3]
                model = ("realtime", "grok-voice", "evi-3")[run_index % 3]
                voice = None if run_index % 5 == 0 else f"voice-{run_index % 4}"
                metric_count = 1 + conversation_index % 3
                metric_payloads: list[tuple[str, float | None, str, str]] = [
                    ("V2V", 180.0 + conversation_index, "milliseconds", "success")
                ]
                if metric_count >= 2:
                    metric_payloads.append(
                        (
                            "InstructionFollowing",
                            float((conversation_index % 2) * 100),
                            "percent",
                            "success",
                        )
                    )
                if metric_count == 3:
                    interrupted = conversation_index % 10 == 0
                    metric_payloads.append(
                        (
                            "InterruptionRate",
                            None if interrupted else round(conversation_index / 10, 2),
                            "per_minute",
                            "failed" if interrupted else "success",
                        )
                    )
                for metric, value, unit, status in metric_payloads:
                    values.append(
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
                            captured_at,
                        )
                    )
            cur.executemany(
                """INSERT INTO benchmarks_v2.results
                   (run_id,provider,model,voice,benchmark,metric_type,metric_value,
                    metric_units,audio_filename,status,error,created_at)
                   VALUES (%s,%s,%s,%s,'S2S',%s,%s,%s,%s,%s,NULL,%s)""",
                values,
            )
            result_count += len(values)
            if (run_index + 1) % 25 == 0:
                conn.commit()
        conn.commit()
        cur.execute("SELECT min(id),max(id) FROM benchmarks_v2.results WHERE benchmark='S2S'")
        bounds = cur.fetchone()
        if bounds is not None:
            min_result_id = int(bounds[0])
            max_result_id = int(bounds[1])
    if min_result_id is None or max_result_id is None:
        raise RuntimeError("benchmark seed produced no S2S results")
    return {
        "runs": runs,
        "conversations": runs * conversations_per_run,
        "results": result_count,
        "min_result_id": min_result_id,
        "max_result_id": max_result_id,
    }


def _peak_rss_kib() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak / 1024) if platform.system() == "Darwin" else int(peak)


def _worker(
    send: Connection,
    database_url: str,
    low: int,
    high: int,
    batch_size: int,
    warmups: int,
    iterations: int,
) -> None:
    try:
        measurements: list[dict[str, Any]] = []
        with psycopg.connect(database_url, cursor_factory=_TimingCursor) as conn:
            for _ in range(warmups):
                backfill(
                    conn,
                    min_result_id=low,
                    max_result_id=high,
                    batch_size=batch_size,
                    apply=False,
                    reporter=ProgressReporter(
                        "dry_run", low, high, batch_size, stream=io.StringIO()
                    ),
                )
            for iteration in range(1, iterations + 1):
                _QUERY_TIMER.reset()
                started = time.monotonic()
                report = backfill(
                    conn,
                    min_result_id=low,
                    max_result_id=high,
                    batch_size=batch_size,
                    apply=False,
                    reporter=ProgressReporter(
                        "dry_run", low, high, batch_size, stream=io.StringIO()
                    ),
                )
                runtime = time.monotonic() - started
                measurements.append(
                    {
                        "batch_size": batch_size,
                        "iteration": iteration,
                        "pages": report["source_pages"],
                        "runs": report["source_runs"],
                        "results": report["source_rows"],
                        "conversations": report["source_groups"],
                        "runtime_seconds": round(runtime, 6),
                        "query_duration_seconds": round(_QUERY_TIMER.seconds, 6),
                        "query_count": _QUERY_TIMER.count,
                        "throughput_runs_per_second": round(report["source_runs"] / runtime, 3),
                        "throughput_results_per_second": round(report["source_rows"] / runtime, 3),
                        "throughput_conversations_per_second": round(
                            report["source_groups"] / runtime, 3
                        ),
                        "peak_rss_kib": _peak_rss_kib(),
                    }
                )
        send.send({"measurements": measurements})
    except BaseException:
        send.send({"error": traceback.format_exc()})
    finally:
        send.close()


def _run_batch_worker(
    database_url: str,
    low: int,
    high: int,
    batch_size: int,
    warmups: int,
    iterations: int,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(send, database_url, low, high, batch_size, warmups, iterations),
    )
    process.start()
    send.close()
    process.join()
    if not receive.poll():
        raise RuntimeError(f"batch worker {batch_size} exited without a result")
    payload = receive.recv()
    receive.close()
    if process.exitcode != 0:
        raise RuntimeError(f"batch worker {batch_size} exited with {process.exitcode}")
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return cast("list[dict[str, Any]]", payload["measurements"])


def _markdown(payload: dict[str, Any]) -> str:
    seed = payload["seed"]
    lines = [
        "# Normalized S2S backfill benchmark",
        "",
        (
            f"Seed: {seed['runs']} runs, {seed['conversations']} conversations, "
            f"{seed['results']} results. Query duration is cumulative client-observed "
            "SQL execution time."
        ),
        "",
        "| batch | iter | pages | runs | results | conversations | runtime s | query s | "
        "queries | runs/s | results/s | conversations/s | peak RSS KiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["measurements"]:
        lines.append(
            "| {batch_size} | {iteration} | {pages} | {runs} | {results} | "
            "{conversations} | {runtime_seconds} | {query_duration_seconds} | "
            "{query_count} | {throughput_runs_per_second} | "
            "{throughput_results_per_second} | {throughput_conversations_per_second} | "
            "{peak_rss_kib} |".format(**row)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin-database-url", required=True)
    parser.add_argument("--runs", type=int, default=450)
    parser.add_argument("--conversations-per-run", type=int, default=50)
    parser.add_argument("--batch-sizes", default=",".join(map(str, _BATCH_SIZES)))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    _validate_local_url(parser, args.admin_database_url)
    if args.runs < 1 or args.conversations_per_run < 1:
        parser.error("--runs and --conversations-per-run must be positive")
    if args.warmups < 0 or args.iterations < 1:
        parser.error("--warmups must be non-negative and --iterations must be positive")
    try:
        batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    except ValueError:
        parser.error("--batch-sizes must be comma-separated integers")
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        parser.error("--batch-sizes must contain positive integers")

    database_name = f"coval_s2s_backfill_bench_{uuid.uuid4().hex}"
    database_url = _database_url(args.admin_database_url, database_name)
    created = False
    try:
        _create_database(args.admin_database_url, database_name)
        created = True
        _migrate(database_url)
        seed = _seed(
            database_url,
            runs=args.runs,
            conversations_per_run=args.conversations_per_run,
        )
        measurements = []
        for batch_size in batch_sizes:
            measurements.extend(
                _run_batch_worker(
                    database_url,
                    seed["min_result_id"],
                    seed["max_result_id"],
                    batch_size,
                    args.warmups,
                    args.iterations,
                )
            )
        payload = {
            "seed": seed,
            "warmups_per_batch": args.warmups,
            "iterations_per_batch": args.iterations,
            "measurements": measurements,
        }
        print(json.dumps(payload, sort_keys=True) if args.format == "json" else _markdown(payload))
    finally:
        if created:
            _drop_database(args.admin_database_url, database_name)


if __name__ == "__main__":
    main()
