# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt screening for arena battle generation — the only guard that prevents spend.

``pii_match`` is local and deliberately narrow: Luhn-checked card numbers and national
id patterns only. Phone numbers, order refs and emails are allowed — the arena's
customer-service and booking domains are full of them, and a blanket digit-run rule
flagged one of the hundred curated example prompts.

``moderation_verdict`` calls OpenAI's ``omni-moderation-latest`` (free) for hate,
sexual, violent and self-harm content. It fails **open**, so ``pii_match`` is the
always-on backstop.
"""

from __future__ import annotations

import re
import unicodedata

import structlog
from openai import AsyncOpenAI

from coval_bench.config import Settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODERATION_MODEL = "omni-moderation-latest"

_CARD_CANDIDATE = re.compile(r"(?:\d[ -]?){12,18}\d")
_NATIONAL_ID = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_client: AsyncOpenAI | None = None


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


def pii_match(prompt: str) -> str | None:
    """Return the kind of personal data found in *prompt*, or ``None`` if clean."""
    text = unicodedata.normalize("NFKC", prompt)
    for candidate in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"[ -]", "", candidate.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return "payment_card"
    if _NATIONAL_ID.search(text):
        return "national_id"
    return None


def _get_client(settings: Settings) -> AsyncOpenAI | None:
    """Return a shared moderation client, or ``None`` when no key is configured."""
    global _client
    if _client is None:
        key = settings.openai_api_key
        if key is None:
            return None
        _client = AsyncOpenAI(api_key=key.get_secret_value())
    return _client


async def moderation_verdict(settings: Settings, prompt: str) -> tuple[bool, dict[str, float]]:
    """Return ``(flagged, category_scores)`` for *prompt*, allowing it on any failure.

    Scores come back even when nothing is flagged so callers can persist them, letting
    thresholds be calibrated later and past battles re-judged without re-calling.
    """
    client = _get_client(settings)
    if client is None:
        logger.warning("arena_moderation_unconfigured")
        return False, {}
    try:
        response = await client.moderations.create(model=MODERATION_MODEL, input=prompt)
    except Exception:
        logger.warning("arena_moderation_unavailable", exc_info=True)
        return False, {}
    result = response.results[0]
    scores = {
        name: float(score)
        for name, score in result.category_scores.model_dump(by_alias=True).items()
    }
    return result.flagged, scores
