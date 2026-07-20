# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot backfill of S2S sample recordings for recent buckets.

The daily fetch tick only samples its newest bucket. This CLI walks the last N
days of already-ingested runs, groups them by timeline bucket, and copies each
bucket's shared-clip recordings into the samples bucket by reusing the live
sampler's :func:`copy_tick_samples`. Idempotent: a bucket whose manifest already
exists is skipped, so re-runs only fill gaps and never reshape a published tick.

Run this AFTER the silence-trim sampler ships, so backfilled recordings come out
trimmed too — a bucket already published fat is not reshaped by a later re-run.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import click
import psycopg
import psycopg.rows
import structlog
from psycopg_pool import AsyncConnectionPool

from coval_bench.config import Settings, get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.s2s.fetch_v2v import _client
from coval_bench.s2s.samples import SampleRun, copy_tick_samples

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = structlog.get_logger("coval_bench.s2s.backfill")

# Newest ingested run per (provider, bucket) over the window. audio_filename is
# "<coval_run_id>/<sim_id>", so its prefix is the Coval run id; scheduled_at is
# the timeline bucket the sampler keys folders by. Failed runs never reached the
# bucket, so only succeeded/partial rows are eligible.
_RUNS_BY_BUCKET_SQL = """
    SELECT DISTINCT ON (r.provider, rn.scheduled_at)
        r.provider AS provider,
        r.model AS model,
        rn.scheduled_at AS scheduled_at,
        split_part(r.audio_filename, '/', 1) AS coval_run_id
    FROM benchmarks_v2.results r
    JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
    WHERE r.benchmark = 'S2S'
      AND rn.status IN ('succeeded', 'partial')
      AND rn.scheduled_at >= %(since)s
      AND r.audio_filename <> ''
    ORDER BY r.provider, rn.scheduled_at, rn.id DESC
"""


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[datetime, list[SampleRun]]:
    """Group query rows into one SampleRun list per bucket timestamp."""
    buckets: dict[datetime, list[SampleRun]] = {}
    for row in rows:
        coval_run_id = cast("str", row["coval_run_id"])
        if not coval_run_id:
            continue
        bucket_at = cast("datetime", row["scheduled_at"])
        buckets.setdefault(bucket_at, []).append(
            SampleRun(
                provider=cast("str", row["provider"]),
                model=cast("str", row["model"]),
                coval_run_id=coval_run_id,
                bucket_at=bucket_at,
            )
        )
    return buckets


async def _runs_by_bucket(
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    since: datetime,
) -> dict[datetime, list[SampleRun]]:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(_RUNS_BY_BUCKET_SQL, {"since": since})
            rows = await cur.fetchall()
        await conn.commit()
    return _group_rows(rows)


async def backfill_v2v_samples(settings: Settings, *, days: int) -> dict[str, int]:
    """Copy sample recordings for every eligible bucket in the last *days*."""
    if not settings.s2s_samples_bucket:
        raise RuntimeError("s2s_samples_bucket is not set")
    since = datetime.now(tz=UTC) - timedelta(days=days)
    stored = 0
    async with _client(settings) as client, lifespan_pool(settings) as pool:
        buckets = await _runs_by_bucket(pool, since)
        # Oldest first so each prepend leaves index.json newest-first.
        for bucket_at in sorted(buckets):
            runs = buckets[bucket_at]
            recordings = await copy_tick_samples(
                client,
                bucket_name=settings.s2s_samples_bucket,
                runs=runs,
                rng=random.Random(),  # noqa: S311 -- sample pick, not security
            )
            if recordings > 0:
                stored += 1
            logger.info(
                "backfill_bucket",
                bucket_at=bucket_at.isoformat(),
                providers=len(runs),
                recordings=recordings,
            )
    result = {"buckets": len(buckets), "stored": stored, "skipped": len(buckets) - stored}
    logger.info("backfill_done", days=days, **result)
    return result


@click.command(name="backfill-s2s")
@click.option(
    "--days",
    default=7,
    show_default=True,
    type=click.IntRange(min=1),
    help="How many days back to backfill sample recordings.",
)
def backfill_s2s(days: int) -> None:
    """One-shot: seed sample recordings for the last N days of ingested runs."""
    from coval_bench.logging import configure_logging

    settings = get_settings()
    configure_logging(level=settings.log_level)
    result = asyncio.run(backfill_v2v_samples(settings, days=days))
    click.echo(json.dumps({"event": "backfill_s2s", **result}))


if __name__ == "__main__":
    backfill_s2s()
