# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot backfill of the WER error-type split from stored transcripts.

Recomputes pre-0014 rows from the stored hypothesis + the packaged manifest
reference. Rows whose recomputed score doesn't match ``metric_value`` (older
normalizer) keep their null split — split-sums-to-total outranks coverage.
STT only: a TTS synth filename doesn't identify the prompt item.
"""

from __future__ import annotations

from pathlib import Path

import click
import psycopg

from coval_bench.config import get_settings
from coval_bench.datasets.loader import _load_manifest
from coval_bench.datasets.manifest import STTManifestItem
from coval_bench.db.writer import STATS_MATVIEWS
from coval_bench.metrics.wer import compute_wer

_SCORE_TOLERANCE = 1e-6
_PAGE_SIZE = 5000

_SELECT_PAGE = """
    SELECT r.id, rn.dataset_id, r.audio_filename, r.transcript, r.metric_value
    FROM benchmarks_v2.results r
    JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
    WHERE r.benchmark = 'STT' AND r.metric_type = 'WER' AND r.status = 'success'
      AND r.transcript IS NOT NULL AND r.metric_value IS NOT NULL
      AND r.wer_insertions_pct IS NULL AND r.id > %s
    ORDER BY r.id
    LIMIT %s
"""

_UPDATE = """
    UPDATE benchmarks_v2.results
    SET wer_insertions_pct = %s, wer_deletions_pct = %s, wer_substitutions_pct = %s
    WHERE id = %s
"""


def _references(dataset_id: str) -> dict[str, str] | None:
    """Map audio basename -> reference transcript, or None for unknown datasets."""
    try:
        manifest = _load_manifest(dataset_id)
    except FileNotFoundError:
        return None
    return {
        Path(item.path).name: item.transcript
        for item in manifest.items
        if isinstance(item, STTManifestItem)
    }


def _split_for_row(
    reference: str | None, transcript: str, metric_value: float
) -> tuple[float, float, float] | None:
    if reference is None:
        return None
    result = compute_wer(reference, transcript)
    if abs(result.wer_percentage - metric_value) > _SCORE_TOLERANCE:
        return None
    split = result.error_percentages
    return (
        split["wer_insertions_pct"],
        split["wer_deletions_pct"],
        split["wer_substitutions_pct"],
    )


def backfill(conn: psycopg.Connection, *, dry_run: bool = False) -> dict[str, int]:
    """Fill the null splits page by page; returns outcome counts."""
    counts = {"updated": 0, "missing_reference": 0, "score_drift": 0}
    lookups: dict[str, dict[str, str] | None] = {}
    last_id = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(_SELECT_PAGE, (last_id, _PAGE_SIZE))
            rows = cur.fetchall()
            if not rows:
                break
            last_id = rows[-1][0]
            updates = []
            for row_id, dataset_id, audio_filename, transcript, metric_value in rows:
                if dataset_id not in lookups:
                    lookups[dataset_id] = _references(dataset_id)
                refs = lookups[dataset_id]
                reference = refs.get(audio_filename) if refs and audio_filename else None
                if reference is None:
                    counts["missing_reference"] += 1
                    continue
                split = _split_for_row(reference, transcript, metric_value)
                if split is None:
                    counts["score_drift"] += 1
                    continue
                updates.append((*split, row_id))
            if updates and not dry_run:
                cur.executemany(_UPDATE, updates)
            counts["updated"] += len(updates)
        conn.commit()
    return counts


def _refresh_matviews(url: str) -> None:
    # CONCURRENTLY refuses to run inside a transaction block, hence autocommit.
    with psycopg.connect(url, autocommit=True) as conn:
        for view in STATS_MATVIEWS:
            conn.execute(  # noqa: S608 — view names are constants
                f"REFRESH MATERIALIZED VIEW CONCURRENTLY benchmarks_v2.{view}"
            )


@click.command()
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Count what would change; no writes, no matview refresh.",
)
def backfill_wer_breakdown_cli(dry_run: bool) -> None:
    """Backfill wer_*_pct on historical STT WER rows from stored transcripts."""
    url = str(get_settings().database_url)
    with psycopg.connect(url) as conn:
        counts = backfill(conn, dry_run=dry_run)
    click.echo(
        f"{'would update' if dry_run else 'updated'} {counts['updated']:,} rows "
        f"(missing reference: {counts['missing_reference']:,}, "
        f"score drift: {counts['score_drift']:,})"
    )
    if counts["updated"] and not dry_run:
        _refresh_matviews(url)
        click.echo(f"refreshed matviews: {', '.join(STATS_MATVIEWS)}")
