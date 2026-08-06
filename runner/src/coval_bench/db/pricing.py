# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``PricingStore`` — access to the append-only ``model_pricing`` ratesheet.

A rate is never UPDATEd: ``upsert_rate`` inserts a new row and stamps
``superseded_at`` on the previous effective one, in a single transaction. That
keeps price history intact and makes historical cost attribution a range query
(``get_effective_rates`` with the run's timestamp).

The orchestrator resolves rates once per result row, so ``load_cache`` pulls
every currently-effective row in one query at run start and
``effective_rates_cached`` serves lookups from memory.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.models import Benchmark, BillingUnit, PriceRow

_COLUMNS = (
    "id, provider, model, benchmark, billing_unit, rate_usd, plan_assumption, "
    "effective_at, superseded_at, source_url, as_of, evidence, updated_by, created_at"
)


class PricingStore:
    """Typed helpers over ``benchmarks_v2.model_pricing``."""

    def __init__(
        self,
        pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    ) -> None:
        self._pool = pool
        self._cache: dict[tuple[str, str], list[PriceRow]] | None = None

    async def get_effective_rates(
        self,
        provider: str,
        model: str,
        at: datetime,
        benchmark: Benchmark | None = None,
    ) -> list[PriceRow]:
        """Rates in force at *at*: ``effective_at <= at < coalesce(superseded_at, ∞)``.

        Token-billed models return two rows (input + output units). Pass
        *benchmark* whenever the caller knows it: (provider, model) alone is
        not unique — gradium serves STT and TTS under one model id, and an
        unfiltered read would hand back incompatible rates together.
        """
        sql = f"""
            SELECT {_COLUMNS}
            FROM benchmarks_v2.model_pricing
            WHERE provider = %(provider)s AND model = %(model)s
              AND (%(benchmark)s::text IS NULL OR benchmark = %(benchmark)s)
              AND effective_at <= %(at)s
              AND (superseded_at IS NULL OR superseded_at > %(at)s)
            ORDER BY billing_unit
        """  # noqa: S608 — column list is a module constant
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    sql,
                    {"provider": provider, "model": model, "at": at, "benchmark": benchmark},
                )
                rows = await cur.fetchall()
            await conn.commit()
        return [PriceRow.model_validate(dict(r)) for r in rows]

    async def upsert_rate(
        self,
        *,
        provider: str,
        model: str,
        benchmark: Benchmark,
        billing_unit: BillingUnit,
        rate_usd: Decimal,
        source_url: str,
        as_of: date,
        updated_by: Literal["human", "bot"],
        plan_assumption: str | None = None,
        evidence: str | None = None,
        effective_at: datetime | None = None,
    ) -> tuple[PriceRow, bool]:
        """Insert a new effective rate, superseding the previous one, atomically.

        Idempotent: when the currently effective row already carries the same
        ``rate_usd`` and ``plan_assumption``, nothing is written and the
        existing row is returned with ``created=False``. Otherwise the old row
        gets ``superseded_at = effective_at`` (contiguous history) and the new
        row is inserted.
        """
        effective_at = effective_at or datetime.now(tz=UTC)
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT {_COLUMNS}
                    FROM benchmarks_v2.model_pricing
                    WHERE provider = %s AND model = %s AND benchmark = %s
                      AND billing_unit = %s AND superseded_at IS NULL
                    FOR UPDATE
                    """,  # noqa: S608 — column list is a module constant
                    (provider, model, benchmark, billing_unit),
                )
                current = await cur.fetchone()
                if (
                    current is not None
                    and current["rate_usd"] == rate_usd
                    and current["plan_assumption"] == plan_assumption
                ):
                    await conn.commit()
                    return PriceRow.model_validate(dict(current)), False

                if current is not None:
                    await cur.execute(
                        "UPDATE benchmarks_v2.model_pricing SET superseded_at = %s WHERE id = %s",
                        (effective_at, current["id"]),
                    )
                await cur.execute(
                    f"""
                    INSERT INTO benchmarks_v2.model_pricing
                        (provider, model, benchmark, billing_unit, rate_usd, plan_assumption,
                         effective_at, source_url, as_of, evidence, updated_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING {_COLUMNS}
                    """,  # noqa: S608 — column list is a module constant
                    (
                        provider,
                        model,
                        benchmark,
                        billing_unit,
                        rate_usd,
                        plan_assumption,
                        effective_at,
                        source_url,
                        as_of,
                        evidence,
                        updated_by,
                    ),
                )
                row = await cur.fetchone()
                if row is None:  # pragma: no cover — unreachable after INSERT RETURNING
                    raise RuntimeError("INSERT INTO model_pricing returned no row")
            await conn.commit()
        return PriceRow.model_validate(dict(row)), True

    async def load_cache(self) -> None:
        """Load every currently-effective rate into memory, one query per run."""
        sql = f"""
            SELECT {_COLUMNS}
            FROM benchmarks_v2.model_pricing
            WHERE superseded_at IS NULL
        """  # noqa: S608 — column list is a module constant
        cache: dict[tuple[str, str], list[PriceRow]] = {}
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(sql)
                rows = await cur.fetchall()
            await conn.commit()
        for r in rows:
            price = PriceRow.model_validate(dict(r))
            cache.setdefault((price.provider, price.model), []).append(price)
        self._cache = cache

    def effective_rates_cached(
        self, provider: str, model: str, benchmark: Benchmark | None = None
    ) -> list[PriceRow]:
        """Currently-effective rates from the in-process cache ([] = unpriced model).

        Same *benchmark* rule as :meth:`get_effective_rates` — pass it when known.
        """
        if self._cache is None:
            raise RuntimeError("PricingStore cache not loaded — call load_cache() first")
        rates = self._cache.get((provider, model), [])
        if benchmark is None:
            return rates
        return [r for r in rates if r.benchmark is benchmark]
