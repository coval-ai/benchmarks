# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot import of past-7-days legacy Neon rows into benchmarks_v2."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Final

import click
import psycopg
import psycopg.rows

from coval_bench.config import get_settings
from coval_bench.runner.config import DEFAULT_STT_MATRIX, DEFAULT_TTS_MATRIX

LEGACY_RUNNER_SHA: Final[str] = "historical-import"
LEGACY_DATASET_ID: Final[str] = "legacy:cv+local"
LEGACY_DATASET_SHA256: Final[str] = "legacy:unverified"
WINDOW_DAYS: Final[int] = 7

# Maps legacy status → (target status, target error).
_LEGACY_STATUS_MAP: Final[dict[str, tuple[str, str | None]]] = {
    "success": ("success", None),
    "tts_failed": ("failed", "legacy_status:tts_failed"),
}


@dataclass(frozen=True)
class LegacyRow:
    provider: str
    model: str
    voice: str | None
    benchmark: str  # "STT" | "TTS"
    metric_type: str
    metric_value: float | None
    metric_units: str | None
    audio_filename: str | None
    transcript: str | None
    timestamp: datetime
    status: str  # "success" | "tts_failed"


def _build_matrix_lookup() -> set[tuple[str, str]]:
    """Return the canonical {(provider_lower, model)} set from both matrices."""
    pairs: set[tuple[str, str]] = set()
    for entry in DEFAULT_STT_MATRIX + DEFAULT_TTS_MATRIX:
        pairs.add((entry.provider.lower(), entry.model))
    return pairs


def _read_legacy(conn: psycopg.Connection[dict[str, object]]) -> list[LegacyRow]:
    """Fetch rows from the legacy table within the 7-day window."""
    cutoff = datetime.now(tz=UTC) - timedelta(days=WINDOW_DAYS)
    sql = """
        SELECT provider, model, voice, benchmark,
               metric_type, metric_value, metric_units,
               audio_filename, transcript, timestamp, status
        FROM public.all_benchmarks
        WHERE timestamp >= %s
        ORDER BY timestamp ASC
    """
    rows: list[LegacyRow] = []
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, (cutoff,))
        for r in cur:
            voice_raw = r["voice"]
            voice: str | None = None if (voice_raw is None or voice_raw == "") else str(voice_raw)

            transcript_raw = r["transcript"]
            transcript: str | None = None if transcript_raw is None else str(transcript_raw)

            metric_value_raw = r["metric_value"]
            metric_value: float | None = (
                float(metric_value_raw) if metric_value_raw is not None else None
            )

            ts_raw = r["timestamp"]
            if isinstance(ts_raw, datetime):
                ts = ts_raw if ts_raw.tzinfo is not None else ts_raw.replace(tzinfo=UTC)
            else:
                raise ValueError(f"unexpected timestamp type: {type(ts_raw)}")

            rows.append(
                LegacyRow(
                    provider=str(r["provider"]),
                    model=str(r["model"]),
                    voice=voice,
                    benchmark=str(r["benchmark"]),
                    metric_type=str(r["metric_type"]),
                    metric_value=metric_value,
                    metric_units=str(r["metric_units"]) if r["metric_units"] is not None else None,
                    audio_filename=(
                        str(r["audio_filename"]) if r["audio_filename"] is not None else None
                    ),
                    transcript=transcript,
                    timestamp=ts,
                    status=str(r["status"]),
                )
            )
    return rows


def _validate(
    rows: Iterable[LegacyRow],
) -> tuple[set[tuple[str, str]], Counter[str]]:
    """Return (unmatched_pairs, status_counts). Pure — no I/O."""
    matrix = _build_matrix_lookup()
    status_counts: Counter[str] = Counter()
    unmatched_counts: Counter[tuple[str, str]] = Counter()

    for row in rows:
        status_counts[row.status] += 1
        key = (row.provider.lower(), row.model)
        if key not in matrix:
            unmatched_counts[key] += 1

    return set(unmatched_counts.keys()), status_counts


def _group_by_day(rows: Iterable[LegacyRow]) -> Mapping[date, list[LegacyRow]]:
    """Group rows by UTC calendar day of timestamp."""
    groups: dict[date, list[LegacyRow]] = defaultdict(list)
    for row in rows:
        day = row.timestamp.astimezone(UTC).date()
        groups[day].append(row)
    return groups


def _redact_url(url: str) -> str:
    """Return host/dbname only — strip credentials from the URL."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or ""
        dbname = parsed.path.lstrip("/")
        return f"{host}/{dbname}"
    except Exception:
        return "<url>"


def _summarize(
    rows: list[LegacyRow],
    unmatched: set[tuple[str, str]],
    status_counts: Counter[str],
) -> str:
    """Return the dry-run report text. Pure — no I/O."""
    matrix = _build_matrix_lookup()

    lines: list[str] = ["== legacy-import dry-run =="]

    legacy_url = os.environ.get("LEGACY_DATABASE_URL", "")
    target_url = ""
    try:
        target_url = str(get_settings().database_url)
    except Exception:
        target_url = "<not configured>"

    lines.append(f"source: legacy import ({_redact_url(legacy_url)})")
    lines.append(f"target: {_redact_url(target_url)}")

    if rows:
        utc_ts = [r.timestamp.astimezone(UTC) for r in rows]
        window_start = min(utc_ts).replace(hour=0, minute=0, second=0, microsecond=0)
        window_end = max(utc_ts)
        lines.append(
            f"window: {window_start.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            f" .. {window_end.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
    else:
        lines.append("window: (no rows)")

    lines.append("")
    lines.append("Row count by (provider, model, metric_type):")

    # Count per (provider, model, metric_type), stable sort
    combo_counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        combo_counts[(row.provider, row.model, row.metric_type)] += 1

    for (provider, model, metric_type), count in sorted(combo_counts.items()):
        lines.append(f"  {provider:<16} {model:<24} {metric_type:<12} {count:>6}")

    lines.append(f"TOTAL: {len(rows):,} rows")
    lines.append("")
    lines.append("Status distribution:")
    for status_val, count in sorted(status_counts.items()):
        lines.append(f"  {status_val:<12} {count:,}")

    groups = _group_by_day(rows)
    lines.append("")
    lines.append("Proposed runs (one per UTC day):")
    for day in sorted(groups):
        day_rows = groups[day]
        day_ts = [r.timestamp.astimezone(UTC) for r in day_rows]
        start = min(day_ts)
        end = max(day_ts)
        lines.append(
            f"  day={day}  rows={len(day_rows)}"
            f"  start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"  end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )

    lines.append("")
    lines.append("Provider/model validation against DEFAULT_STT_MATRIX + DEFAULT_TTS_MATRIX:")

    # Count combos per provider_lower/model
    all_combos: set[tuple[str, str]] = set()
    unmatched_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (row.provider.lower(), row.model)
        all_combos.add(key)
        if key not in matrix:
            unmatched_counts[key] += 1

    matched_count = len(all_combos) - len(unmatched)
    lines.append(f"  matched   : {matched_count} combos")
    lines.append(f"  unmatched : {len(unmatched)} combos")
    for key in sorted(unmatched):
        count = unmatched_counts[key]
        lines.append(f"    - ({key[0]}, {key[1]})     row_count={count:>4}")

    lines.append("")
    if unmatched:
        skipped_rows = sum(unmatched_counts.values())
        lines.append(
            "Dry-run complete. No writes performed.\n"
            f"Real import will SKIP the {len(unmatched)} unmatched combo(s)"
            f" ({skipped_rows:,} rows) and proceed with the rest."
        )
    else:
        lines.append("Dry-run complete. No writes performed.")

    return "\n".join(lines)


def _execute_import(target_url: str, rows: list[LegacyRow]) -> None:
    """DELETE existing legacy synthetic runs, INSERT new ones + their results.

    Wraps the DELETE and all INSERTs in a single transaction — if any INSERT
    fails the DELETE rolls back, leaving the target in its prior state.
    """
    groups = _group_by_day(rows)

    insert_run_sql = """
        INSERT INTO benchmarks_v2.runs
            (runner_sha, dataset_id, dataset_sha256,
             started_at, finished_at, status, error)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """

    # created_at is overridden here — the DB default (now()) must NOT be used.
    # See ADR-003: per-row timestamps drive the front-end 15-min bucketing.
    insert_result_sql = """
        INSERT INTO benchmarks_v2.results
            (run_id, provider, model, voice, benchmark,
             metric_type, metric_value, metric_units,
             audio_filename, transcript, status, error, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    with psycopg.connect(target_url) as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM benchmarks_v2.runs WHERE runner_sha = %s",
            (LEGACY_RUNNER_SHA,),
        )

        for day in sorted(groups):
            day_rows = groups[day]
            day_ts = [r.timestamp.astimezone(UTC) for r in day_rows]
            started_at = min(day_ts)
            finished_at = max(day_ts)

            cur.execute(
                insert_run_sql,
                (
                    LEGACY_RUNNER_SHA,
                    LEGACY_DATASET_ID,
                    LEGACY_DATASET_SHA256,
                    started_at,
                    finished_at,
                    "succeeded",
                    None,
                ),
            )
            run_row = cur.fetchone()
            if run_row is None:  # pragma: no cover
                raise RuntimeError("INSERT INTO runs returned no row")
            run_id: int = run_row[0]

            result_params = []
            for row in day_rows:
                target_status, error = _LEGACY_STATUS_MAP[row.status]
                result_params.append(
                    (
                        run_id,
                        row.provider.lower(),
                        row.model,
                        row.voice,
                        row.benchmark,
                        row.metric_type,
                        row.metric_value,
                        row.metric_units,
                        row.audio_filename,
                        row.transcript,
                        target_status,
                        error,
                        row.timestamp.astimezone(UTC),
                    )
                )
            cur.executemany(insert_result_sql, result_params)

        conn.commit()
        cur.close()


def map_status(legacy_status: str) -> tuple[str, str | None]:
    """Map a legacy status string to (target_status, error). Pure."""
    return _LEGACY_STATUS_MAP[legacy_status]


@click.command()
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Print row counts per (provider, model, metric_type) and proposed run grouping; no writes."
    ),
)
def import_legacy_cli(dry_run: bool) -> None:
    """Import past 7 days of legacy benchmarks into benchmarks_v2.runs."""
    legacy_url = os.environ.get("LEGACY_DATABASE_URL")
    if not legacy_url:
        raise click.ClickException("LEGACY_DATABASE_URL not set")

    target_url = str(get_settings().database_url)

    with psycopg.connect(legacy_url, row_factory=psycopg.rows.dict_row) as conn:
        rows = _read_legacy(conn)

    unmatched, status_counts = _validate(rows)
    click.echo(_summarize(rows, unmatched, status_counts))

    if dry_run:
        return

    expected_statuses = {"success", "tts_failed"}
    unexpected = set(status_counts) - expected_statuses
    if unexpected:
        click.echo(f"ERROR: unexpected legacy status values: {sorted(unexpected)}", err=True)
        raise SystemExit(2)

    if unmatched:
        rows_to_import = [r for r in rows if (r.provider.lower(), r.model) not in unmatched]
        skipped = len(rows) - len(rows_to_import)
        click.echo(
            f"Skipping {skipped:,} rows from {len(unmatched)} unmatched combo(s);"
            f" importing {len(rows_to_import):,} rows.",
            err=True,
        )
    else:
        rows_to_import = list(rows)

    _execute_import(target_url, rows_to_import)
    click.echo(f"Imported {len(rows_to_import):,} rows into benchmarks_v2.results.")


if __name__ == "__main__":
    import_legacy_cli()
