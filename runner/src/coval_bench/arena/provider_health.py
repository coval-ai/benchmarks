# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Which arena providers cannot serve: no key configured here, or the last benchmark run
proved the key dead.

The TTS benchmark pays real credits against every provider every 30 minutes, so
``benchmarks_v2.results`` already answers the second question. Reading it keeps the arena
free of health state of its own, and of the races that come with sharing state across
instances. A provider returns to pairing when a later run synthesizes for it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

import psycopg.rows
import structlog
from psycopg_pool import AsyncConnectionPool

from coval_bench.config import Settings
from coval_bench.registries.provider_keys import PROVIDER_ENV

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Every TTFA row of each provider's most recent TTS run. TTFA only: a run also writes WER
# and latency-breakdown rows, which say nothing about whether the key works. The whole
# run rather than its last row: a provider can field several models, they finish out of
# order, and one arbitrary row must not decide the provider's health.
_LATEST_RUN_TTFA_SQL = """
    WITH latest_run AS (
        SELECT DISTINCT ON (r.provider) r.provider, r.run_id
        FROM benchmarks_v2.results r
        JOIN benchmarks_v2.runs u ON u.id = r.run_id
        WHERE r.benchmark = 'TTS'
          AND r.metric_type = 'TTFA'
          AND u.started_at > now() - interval '1 day'
        ORDER BY r.provider, u.started_at DESC, r.run_id DESC
    )
    SELECT l.provider, r.status, r.error
    FROM latest_run l
    JOIN benchmarks_v2.results r
      ON r.run_id = l.run_id AND r.provider = l.provider
    WHERE r.benchmark = 'TTS' AND r.metric_type = 'TTFA'
"""


class KeyFailure(StrEnum):
    """Why a provider call failed in a way that implicates its key."""

    CREDIT = "credit"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"


# Read when the provider surfaced no status code — websocket and SDK integrations report
# prose.
_CREDIT_PHRASES = (
    "insufficient",
    "out of credit",
    "credit balance",
    "quota exceeded",
    "exceeded your quota",
    "usage limit",
    "payment required",
    "billing",
    "subscription",
    "upgrade your plan",
)
_AUTH_PHRASES = (
    "unauthorized",
    "unauthenticated",
    "invalid api key",
    "invalid_api_key",
    "api key not valid",
    "forbidden",
    "authentication failed",
)
_RATE_LIMIT_PHRASES = ("rate limit", "rate_limit", "too many requests")

_STATUS_REASONS = {401: KeyFailure.AUTH, 402: KeyFailure.CREDIT, 403: KeyFailure.AUTH}

# 21 of the 26 TTS integrations speak websocket and carry no status object; a rejected
# handshake arrives as "server rejected WebSocket connection: HTTP 401".
_EMBEDDED_STATUS = re.compile(r"\bHTTP[ /]?(\d{3})\b", re.IGNORECASE)

# A rate limit resolves on its own, so it costs a swap, not a bench.
BENCHING_REASONS = frozenset({KeyFailure.CREDIT, KeyFailure.AUTH})


def classify_failure(status_code: int | None, error: str | None) -> KeyFailure | None:
    """Why this failure implicates the key, or None if it does not.

    The status code decides when there is one; 26 integrations word their errors
    differently.
    """
    if status_code is None:
        status_code = _status_in_prose(error)
    if status_code is not None:
        reason = _STATUS_REASONS.get(status_code)
        if reason is not None:
            return reason
        # A 429 naming quota is exhaustion wearing a rate limit's status.
        if status_code == 429:
            return KeyFailure.CREDIT if _mentions(error, _CREDIT_PHRASES) else KeyFailure.RATE_LIMIT
        return None
    if _mentions(error, _CREDIT_PHRASES):
        return KeyFailure.CREDIT
    if _mentions(error, _AUTH_PHRASES):
        return KeyFailure.AUTH
    if _mentions(error, _RATE_LIMIT_PHRASES):
        return KeyFailure.RATE_LIMIT
    return None


def _status_in_prose(error: str | None) -> int | None:
    """An HTTP status quoted inside an error string, when the provider reported none."""
    if not error:
        return None
    found = _EMBEDDED_STATUS.search(error)
    return int(found.group(1)) if found is not None else None


def _mentions(error: str | None, phrases: tuple[str, ...]) -> bool:
    if not error:
        return False
    haystack = error.lower()
    return any(phrase in haystack for phrase in phrases)


def unconfigured_providers(settings: Settings, providers: Iterable[str]) -> frozenset[str]:
    """Roster providers with no API key on this service.

    Eligibility, not a failure classification: such a provider fails every call, so it
    leaves pairing rather than being swapped on every request. The benchmark cannot
    answer this — the job mounts its keys separately from this service. Providers with no
    ``PROVIDER_ENV`` entry authenticate another way and are never excluded.
    """
    return frozenset(
        provider
        for provider in providers
        if (env_var := PROVIDER_ENV.get(provider)) is not None
        and getattr(settings, env_var.lower(), None) is None
    )


async def benchmark_benched_providers(pool: AsyncConnectionPool[Any]) -> frozenset[str]:
    """Providers whose most recent TTS run produced no audio and a dead-key error.

    One success anywhere in that run clears the provider: a key that synthesized is a key
    that works, whatever else the run reported.

    Fails open on a read error — an unreadable table costs a swapped battle, raising
    would cost every one.
    """
    try:
        async with pool.connection() as conn:
            conn.row_factory = psycopg.rows.dict_row
            cursor = await conn.execute(_LATEST_RUN_TTFA_SQL)
            rows = await cursor.fetchall()
    except Exception:
        logger.warning("arena_provider_health_read_failed", exc_info=True)
        return frozenset()

    succeeded: set[str] = set()
    failed_on_key: set[str] = set()
    for row in rows:
        if row["status"] == "success":
            succeeded.add(row["provider"])
        elif classify_failure(None, row["error"]) in BENCHING_REASONS:
            failed_on_key.add(row["provider"])
    return frozenset(failed_on_key - succeeded)


def report_key_failure(
    *, provider: str, model: str, reason: KeyFailure, status_code: int | None
) -> None:
    """Announce a dead key the arena hit itself — the log line the alert is built on.

    Carries no provider body on purpose: the reason and status are what an alert acts on,
    and a raw upstream error is where a leaked credential would travel. Failures the
    *benchmark* discovers are not announced here — that is the benchmark's alert to raise.
    """
    logger.error(
        "arena_provider_key_exhausted",
        provider=provider,
        model=model,
        reason=reason.value,
        status_code=status_code,
    )
