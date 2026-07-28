# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt screening for arena battle generation — the only guard that prevents spend.

``pii_match``: local and narrow — national ids, and cards with an issuer prefix that also
pass Luhn. Phone numbers and order refs are allowed; a blanket digit rule rejected one of
the curated example prompts.

``moderation_verdict``: OpenAI ``omni-moderation-latest`` (free). Reports whether the
service answered; the caller decides what silence means. Bounded by our own timeout as
well as the client's, since the public route dies at 60s and the SDK defaults to 600.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

import structlog
from openai import AsyncOpenAI

from coval_bench.config import Settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODERATION_MODEL = "omni-moderation-latest"
MODERATION_TIMEOUT_S = 4.0

_CARD_CANDIDATE = re.compile(r"(?:\d[ -]?){12,18}\d")
_NATIONAL_ID = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_client: AsyncOpenAI | None = None
_client_fingerprint: str | None = None


@dataclass(frozen=True)
class ModerationResult:
    """What the moderator said, and whether it answered at all."""

    flagged: bool
    available: bool
    scores: dict[str, float] = field(default_factory=dict)


def _luhn_ok(digits: str) -> bool:
    """Whether *digits* satisfies the Luhn checksum used by payment cards."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _has_card_iin(digits: str) -> bool:
    """Whether *digits* opens with a known card issuer prefix.

    Luhn alone passes ~1 in 10 arbitrary numbers, so a tracking reference of the right
    length would read as a card without this.
    """
    two, three, four = int(digits[:2]), int(digits[:3]), int(digits[:4])
    return (
        digits[0] == "4"
        or two in {34, 36, 37, 38, 39, 65}
        or 51 <= two <= 55
        or 2221 <= four <= 2720
        or 3528 <= four <= 3589
        or 300 <= three <= 305
        or 644 <= three <= 649
        or digits[:4] == "6011"
    )


def pii_match(prompt: str) -> str | None:
    """Return the kind of personal data found in *prompt*, or ``None`` if clean."""
    text = unicodedata.normalize("NFKC", prompt)
    for candidate in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"[ -]", "", candidate.group())
        if 13 <= len(digits) <= 19 and _has_card_iin(digits) and _luhn_ok(digits):
            return "payment_card"
    if _NATIONAL_ID.search(text):
        return "national_id"
    return None


async def _get_client(settings: Settings) -> AsyncOpenAI | None:
    """Return a client for the configured key, rebuilding it when the key changes.

    The superseded client is closed, otherwise each rotation leaks its connection pool.
    """
    global _client, _client_fingerprint
    key = settings.openai_api_key
    if key is None:
        return None
    secret = key.get_secret_value()
    fingerprint = hashlib.sha256(secret.encode()).hexdigest()
    if _client is None or _client_fingerprint != fingerprint:
        if _client is not None:
            await _client.close()
        _client = AsyncOpenAI(
            api_key=secret,
            timeout=MODERATION_TIMEOUT_S,
            max_retries=0,
        )
        _client_fingerprint = fingerprint
    return _client


async def moderation_verdict(settings: Settings, prompt: str) -> ModerationResult:
    """Classify *prompt*, reporting ``available=False`` if the moderator did not answer.

    Scores come back even when nothing is flagged, so they can be stored and re-judged
    later without re-calling. Parsing sits inside the guard: a malformed response is an
    unavailable moderator, not a 500.
    """
    client = await _get_client(settings)
    if client is None:
        logger.warning("arena_moderation_unconfigured")
        return ModerationResult(flagged=False, available=False)
    try:
        async with asyncio.timeout(MODERATION_TIMEOUT_S):
            response = await client.moderations.create(model=MODERATION_MODEL, input=prompt)
        result = response.results[0]
        scores = {
            name: float(score)
            for name, score in result.category_scores.model_dump(by_alias=True).items()
        }
        return ModerationResult(flagged=result.flagged, available=True, scores=scores)
    except Exception:
        logger.warning("arena_moderation_unavailable", exc_info=True)
        return ModerationResult(flagged=False, available=False)
