# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``PricingStore``: parameterised reads of the pricing log and its one append door."""

from __future__ import annotations

from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from coval_bench.registries.pricing import NewRate, RateKey, RateRecording

PricingPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

_COLUMNS = (
    "id, benchmark, provider, model, unit, price_usd, effective_from, source_url, notes,"
    " recorded_by_user_id, recorded_by_email, recorded_at"
)
_SELECT = f"SELECT {_COLUMNS} FROM benchmarks_v2.pricing_rates"  # noqa: S608 — constant columns
_BY_KEY = " WHERE benchmark = %s AND provider = %s AND model = %s"
_ORDER = " ORDER BY benchmark, provider, model, effective_from, recorded_at, id"
_INSERT = f"""
    INSERT INTO benchmarks_v2.pricing_rates
        (benchmark, provider, model, unit, price_usd, effective_from, source_url, notes,
         recorded_by_user_id, recorded_by_email)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING {_COLUMNS}
"""  # noqa: S608 — constant columns


def _key_params(key: RateKey) -> tuple[str, str, str]:
    return (key[0].value, key[1], key[2])


class PricingStore:
    """Per-pool persistence helper for the pricing log."""

    def __init__(self, pool: PricingPool) -> None:
        self._pool = pool

    async def _fetch(self, sql: str, params: tuple[Any, ...] = ()) -> list[RateRecording]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        return [RateRecording.model_validate(dict(row)) for row in rows]

    async def recordings(self) -> list[RateRecording]:
        """Every recording ever made, in key then chronological order."""
        return await self._fetch(_SELECT + _ORDER)

    async def recordings_for(self, key: RateKey) -> list[RateRecording]:
        return await self._fetch(_SELECT + _BY_KEY + _ORDER, _key_params(key))

    async def count_for(self, key: RateKey) -> int:
        sql = "SELECT count(*) AS n FROM benchmarks_v2.pricing_rates" + _BY_KEY  # noqa: S608
        async with self._pool.connection() as conn:
            row = await (await conn.execute(sql, _key_params(key))).fetchone()
        return 0 if row is None else int(row["n"])

    async def model_exists(self, key: RateKey) -> bool:
        """Whether the registry lists the model in any state — hidden ones may be priced."""
        sql = "SELECT 1 FROM benchmarks_v2.models" + _BY_KEY.replace(  # noqa: S608
            "benchmark =", "modality ="
        )
        async with self._pool.connection() as conn:
            row = await (await conn.execute(sql, _key_params(key))).fetchone()
        return row is not None

    async def record(
        self, new: NewRate, *, user_id: str, email: str | None
    ) -> tuple[RateRecording, bool]:
        """Append *new* unless it repeats what already describes that date.

        Returns the recording now describing ``new.effective_from`` and whether this
        call wrote it. Writers for one model serialise on an advisory lock so two
        admins saving at once cannot both see an empty date and both append.
        """
        latest_sql = (
            _SELECT
            + _BY_KEY
            + " AND effective_from = %s ORDER BY recorded_at DESC, id DESC LIMIT 1"
        )
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"pricing:{'/'.join(_key_params(new.key))}",),
            )
            latest = await (
                await conn.execute(latest_sql, (*_key_params(new.key), new.effective_from))
            ).fetchone()
            if latest is not None and new.matches(
                existing := RateRecording.model_validate(dict(latest))
            ):
                return existing, False
            params = (
                *_key_params(new.key),
                None if new.unit is None else str(new.unit),
                new.price_usd,
                new.effective_from,
                None if new.source_url is None else str(new.source_url),
                new.notes,
                user_id,
                email,
            )
            row = await (await conn.execute(_INSERT, params)).fetchone()
            if row is None:  # pragma: no cover — unreachable after INSERT RETURNING
                raise RuntimeError("INSERT INTO benchmarks_v2.pricing_rates returned no row")
            return RateRecording.model_validate(dict(row)), True
