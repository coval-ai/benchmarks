# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the packaged pricing registry into the ``pricing_rates`` change-log.

The registry files are the source of truth for what a model costs today; this
log is how cost stays trackable over time. Each sync inserts every packaged
rate and lets the table's unique key drop the ones already recorded, so an
unchanged registry is a no-op and a rate change (new price or new effective
date) appends exactly one row. Append-only: nothing here updates or deletes.
"""

from __future__ import annotations

import click
import psycopg

from coval_bench.registries.pricing import PRICING

_INSERT_SQL = """
    INSERT INTO benchmarks_v2.pricing_rates
        (benchmark, provider, model, unit, price_usd, effective_from,
         source_url, notes, recorded_by)
    VALUES
        (%(benchmark)s, %(provider)s, %(model)s, %(unit)s, %(price_usd)s,
         %(effective_from)s, %(source_url)s, %(notes)s, %(recorded_by)s)
    ON CONFLICT (benchmark, provider, model, unit, price_usd, effective_from)
        DO NOTHING
"""


def sync_pricing_log(conn: psycopg.Connection, recorded_by: str = "pricing-sync") -> int:
    """Append every packaged rate not already recorded; return the rows added."""
    inserted = 0
    for entry in (PRICING[key] for key in sorted(PRICING)):
        cursor = conn.execute(
            _INSERT_SQL,
            {
                "benchmark": entry.benchmark.value,
                "provider": entry.provider,
                "model": entry.model,
                "unit": str(entry.unit),
                "price_usd": entry.price_usd,
                "effective_from": entry.effective_from,
                "source_url": str(entry.source_url),
                "notes": entry.notes,
                "recorded_by": recorded_by,
            },
        )
        inserted += cursor.rowcount
    return inserted


@click.command(name="sync")
def pricing_sync() -> None:
    """Record the packaged ratesheets into the pricing_rates change-log."""
    from coval_bench.config import get_settings

    with psycopg.connect(str(get_settings().database_url)) as conn:
        inserted = sync_pricing_log(conn)
    click.echo(f"pricing sync: {inserted} new rate(s) recorded")
