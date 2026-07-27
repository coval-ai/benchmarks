# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for arena prompt screening (local PII match + moderation verdict)."""

from __future__ import annotations

from typing import Any

import pytest

from coval_bench.arena import moderation
from coval_bench.arena.moderation import moderation_verdict, pii_match
from coval_bench.config import Settings


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Please read back 4242 4242 4242 4242 to confirm.", "payment_card"),
        ("Card 4242-4242-4242-4242 on file.", "payment_card"),
        ("My ssn is 123-45-6789 for the file.", "national_id"),
        # Sixteen digits that fail Luhn are an order reference, not a card.
        ("Your order reference is 1234567890123456, thanks.", None),
        # The curated example prompt that a blanket digit-run rule rejected.
        ("Call us back at 1-800-555-0142, extension 230, if anything changes.", None),
        ("Booking ref 998812 for Tuesday at 4.", None),
        ("Email support@acme.com if it changes.", None),
        ("", None),
    ],
)
def test_pii_match(prompt: str, expected: str | None) -> None:
    """Only Luhn-valid cards and national ids are treated as personal data."""
    assert pii_match(prompt) == expected


def test_example_prompts_are_never_rejected() -> None:
    """The bank must survive its own screening; read off the router because
    ``arena.prompts`` imports ``api.schemas`` and cycles if imported first."""
    from coval_bench.api.routers.arena import EXAMPLE_PROMPTS

    offenders = [p for prompts in EXAMPLE_PROMPTS.values() for p in prompts if pii_match(p)]
    assert offenders == []


async def test_moderation_verdict_allows_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no API key the prompt is allowed and no client is built."""
    monkeypatch.setattr(moderation, "_client", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    flagged, scores = await moderation_verdict(Settings(), "hello")
    assert flagged is False
    assert scores == {}


async def test_moderation_verdict_fails_open_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A moderation outage allows the prompt rather than taking the arena down."""

    class _Boom:
        class moderations:  # noqa: N801
            @staticmethod
            async def create(**_: Any) -> Any:
                raise RuntimeError("upstream down")

    monkeypatch.setattr(moderation, "_client", _Boom())
    flagged, scores = await moderation_verdict(Settings(), "hello")
    assert flagged is False
    assert scores == {}


async def test_moderation_verdict_returns_scores_when_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flagged prompt reports its category scores so callers can persist them."""

    class _Scores:
        @staticmethod
        def model_dump(*, by_alias: bool) -> dict[str, float]:
            assert by_alias is True
            return {"hate": 0.91, "sexual/minors": 0.0}

    class _Result:
        flagged = True
        category_scores = _Scores()

    class _Response:
        results = [_Result()]

    class _Stub:
        class moderations:  # noqa: N801
            @staticmethod
            async def create(**_: Any) -> Any:
                return _Response()

    monkeypatch.setattr(moderation, "_client", _Stub())
    flagged, scores = await moderation_verdict(Settings(), "something vile")
    assert flagged is True
    assert scores["hate"] == pytest.approx(0.91)
