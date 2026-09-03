# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``PricingStore`` — the append-only pricing log.

Mirrors ``db/registry_store.py``: async, the shared psycopg pool, every
statement parameterised. Nothing here updates or deletes; a rate is changed by
recording a new one, and ``registries.pricing.resolve`` decides which recording
describes which day. :meth:`PricingStore.record` is the one door: a recording
identical to the one currently describing that effective date is a no-op (a
double-submit writes nothing), while re-recording a value that lost to a later
correction is a real change and appends.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.pricing.resolve import RateKey, RateRecording
from coval_bench.registries.pricing.schema import PricingEntry, PricingUnit

PricingPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]


class NewRate(BaseModel, frozen=True, extra="forbid"):
    """A rate as an admin states it, before the log assigns identity and time.

    A priced rate passes through :class:`PricingEntry`, so it obeys exactly the
    rules a ratesheet entry does — unit bills the benchmark, price above zero
    and written as a decimal string, a real URL. An unpriced one (``unit`` and
    ``price_usd`` both None) says the model has no known public rate from
    ``effective_from``; it may cite where that was checked but need not.
    """

    benchmark: Benchmark
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    unit: PricingUnit | None = None
    price_usd: Decimal | None = None
    effective_from: date
    source_url: HttpUrl | None = None
    notes: str | None = None

    @field_validator("price_usd", mode="before")
    @classmethod
    def _price_written_as_string(cls, value: object) -> object:
        if isinstance(value, float | int):
            raise ValueError('write price_usd as a string so the decimal is exact, e.g. "0.20"')
        return value

    @field_validator("notes", mode="before")
    @classmethod
    def _blank_notes_are_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _priced_or_delisted(self) -> NewRate:
        if (self.unit is None) != (self.price_usd is None):
            raise ValueError(
                "a rate needs both a unit and a price; "
                "a delisting (no known public rate) has neither"
            )
        if self.price_usd is not None:
            if self.source_url is None:
                raise ValueError("a priced rate must cite the public page that prints the figure")
            # The ratesheet rules, applied to the admin's entry verbatim.
            PricingEntry(
                benchmark=self.benchmark,
                provider=self.provider,
                model=self.model,
                unit=self.unit,
                price_usd=self.price_usd,
                effective_from=self.effective_from,
                source_url=self.source_url,
                notes=self.notes,
            )
        return self

    @property
    def key(self) -> RateKey:
        return (self.benchmark, self.provider, self.model)

    def matches(self, recording: RateRecording) -> bool:
        """Whether *recording* already states this rate for this date.

        Prices compare as written, not as numbers: ``0.20`` and ``0.2`` are the
        same amount but not the same quote, and the site prints the quote.
        """
        return (
            self.key == recording.key
            and self.effective_from == recording.effective_from
            and self.unit == recording.unit
            and _as_written(self.price_usd) == _as_written(recording.price_usd)
            and (None if self.source_url is None else str(self.source_url)) == recording.source_url
            and self.notes == recording.notes
        )


def _as_written(price: Decimal | None) -> str | None:
    return None if price is None else str(price)


_COLUMNS = (
    "id, benchmark, provider, model, unit, price_usd, effective_from, source_url, notes,"
    " recorded_by_user_id, recorded_by_email, recorded_at"
)
_SELECT = f"SELECT {_COLUMNS} FROM benchmarks_v2.pricing_rates"  # noqa: S608 — column list is a constant
_ORDER = " ORDER BY benchmark, provider, model, effective_from, recorded_at, id"

_INSERT = f"""
    INSERT INTO benchmarks_v2.pricing_rates
        (benchmark, provider, model, unit, price_usd, effective_from, source_url, notes,
         recorded_by_user_id, recorded_by_email)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING {_COLUMNS}
"""  # noqa: S608 — column list is a constant

# A recording already in the log with exactly these values, at any time.
_EXISTS_EVER = """
    SELECT 1 FROM benchmarks_v2.pricing_rates
    WHERE benchmark = %s AND provider = %s AND model = %s AND effective_from = %s
      AND unit IS NOT DISTINCT FROM %s
      AND price_usd::text IS NOT DISTINCT FROM %s
      AND source_url IS NOT DISTINCT FROM %s
      AND notes IS NOT DISTINCT FROM %s
    LIMIT 1
"""


def _recording(row: Mapping[str, Any]) -> RateRecording:
    return RateRecording.model_validate(dict(row))


def _insert_params(new: NewRate, user_id: str, email: str | None) -> tuple[Any, ...]:
    return (
        new.benchmark.value,
        new.provider,
        new.model,
        None if new.unit is None else str(new.unit),
        new.price_usd,
        new.effective_from,
        None if new.source_url is None else str(new.source_url),
        new.notes,
        user_id,
        email,
    )


class PricingStore:
    """Per-pool persistence helper for the pricing log."""

    def __init__(self, pool: PricingPool) -> None:
        self._pool = pool

    async def recordings(self) -> list[RateRecording]:
        """Every recording ever made, in key then chronological order."""
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(_SELECT + _ORDER)).fetchall()
        return [_recording(row) for row in rows]

    async def recordings_for(self, key: RateKey) -> list[RateRecording]:
        sql = _SELECT + " WHERE benchmark = %s AND provider = %s AND model = %s" + _ORDER
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, (key[0].value, key[1], key[2]))).fetchall()
        return [_recording(row) for row in rows]

    async def count_for(self, key: RateKey) -> int:
        sql = (
            "SELECT count(*) AS n FROM benchmarks_v2.pricing_rates"
            " WHERE benchmark = %s AND provider = %s AND model = %s"
        )
        async with self._pool.connection() as conn:
            row = await (await conn.execute(sql, (key[0].value, key[1], key[2]))).fetchone()
        return 0 if row is None else int(row["n"])

    async def model_exists(self, key: RateKey) -> bool:
        """Whether the registry lists the model, in any state — hidden ones may be priced."""
        sql = (
            "SELECT 1 FROM benchmarks_v2.models"
            " WHERE modality = %s AND provider = %s AND model = %s"
        )
        async with self._pool.connection() as conn:
            row = await (await conn.execute(sql, (key[0].value, key[1], key[2]))).fetchone()
        return row is not None

    async def record(
        self, new: NewRate, *, user_id: str, email: str | None
    ) -> tuple[RateRecording, bool]:
        """Append *new* unless it repeats what already describes that date.

        Returns the recording now describing ``new.effective_from`` and whether
        this call wrote it. Writers for one model are serialised on an advisory
        lock so two admins saving at once cannot both see an empty date and
        both append.
        """
        lock_key = f"pricing:{new.benchmark.value}:{new.provider}:{new.model}"
        latest_sql = (
            _SELECT
            + " WHERE benchmark = %s AND provider = %s AND model = %s AND effective_from = %s"
            + " ORDER BY recorded_at DESC, id DESC LIMIT 1"
        )
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,))
            latest = await (
                await conn.execute(
                    latest_sql,
                    (new.benchmark.value, new.provider, new.model, new.effective_from),
                )
            ).fetchone()
            if latest is not None:
                existing = _recording(latest)
                if new.matches(existing):
                    return existing, False
            row = await (
                await conn.execute(_INSERT, _insert_params(new, user_id, email))
            ).fetchone()
            if row is None:  # pragma: no cover — unreachable after INSERT RETURNING
                raise RuntimeError("INSERT INTO benchmarks_v2.pricing_rates returned no row")
            return _recording(row), True
