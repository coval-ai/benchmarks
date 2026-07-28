# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for arena prompt screening (local PII match + moderation verdict)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from coval_bench.arena import moderation
from coval_bench.arena.moderation import MODERATION_TIMEOUT_S, moderation_verdict, pii_match
from coval_bench.arena.prompts import EXAMPLE_PROMPTS
from coval_bench.config import Settings


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Please read back 4242 4242 4242 4242 to confirm.", "payment_card"),
        ("Card 4242-4242-4242-4242 on file.", "payment_card"),
        ("Amex 3782 822463 10005 on file.", "payment_card"),
        ("My ssn is 123-45-6789 for the file.", "national_id"),
        # Luhn-valid but no issuer prefix: a tracking reference, not a card.
        ("Tracking 9999999999999995 is out for delivery.", None),
        # Right length and issuer prefix but fails Luhn.
        ("Your order reference is 4242424242424243, thanks.", None),
        # The curated example prompt that a blanket digit-run rule rejected.
        ("Call us back at 1-800-555-0142, extension 230, if anything changes.", None),
        ("Booking ref 998812 for Tuesday at 4.", None),
        ("Email support@acme.com if it changes.", None),
        ("", None),
    ],
)
def test_pii_match(prompt: str, expected: str | None) -> None:
    """Only issuer-prefixed Luhn-valid cards and national ids count as personal data."""
    assert pii_match(prompt) == expected


def test_example_prompts_are_never_rejected() -> None:
    """The bank must survive its own screening."""
    offenders = [p for prompts in EXAMPLE_PROMPTS.values() for p in prompts if pii_match(p)]
    assert offenders == []


def _install_stub(monkeypatch: pytest.MonkeyPatch, create: Any) -> None:
    """Point the module at a stand-in client whose ``moderations.create`` is *create*."""

    class _Stub:
        class moderations:  # noqa: N801
            pass

    _Stub.moderations.create = staticmethod(create)  # type: ignore[attr-defined]

    async def _stub_get_client(_settings: Any) -> Any:
        return _Stub()

    monkeypatch.setattr(moderation, "_get_client", _stub_get_client)


async def test_verdict_unavailable_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no API key the moderator is unavailable, and no client is built.

    ``_env_file=None`` matters: ``Settings`` otherwise reads ``.env``, so a key sitting
    there would build a real client and send this prompt to OpenAI.
    """
    monkeypatch.setattr(moderation, "_client", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    verdict = await moderation_verdict(Settings(_env_file=None), "hello")
    assert verdict.available is False
    assert verdict.flagged is False


@pytest.mark.parametrize(
    "error",
    [RuntimeError("upstream down"), TimeoutError(), ValueError("429 rate limited")],
)
async def test_verdict_unavailable_on_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Any upstream failure — outage, timeout, 429 — reports unavailable, never raises."""

    async def _raise(**_: Any) -> Any:
        raise error

    _install_stub(monkeypatch, _raise)
    verdict = await moderation_verdict(Settings(), "hello")
    assert verdict.available is False
    assert verdict.flagged is False


async def test_verdict_unavailable_on_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty results list is an unavailable moderator, not an unhandled 500."""

    class _Empty:
        results: list[Any] = []

    async def _empty(**_: Any) -> Any:
        return _Empty()

    _install_stub(monkeypatch, _empty)
    verdict = await moderation_verdict(Settings(), "hello")
    assert verdict.available is False


async def test_verdict_returns_scores_when_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flagged prompt reports its category scores so callers can persist them."""

    class _Scores:
        @staticmethod
        def model_dump(*, by_alias: bool) -> dict[str, float]:
            assert by_alias is True
            return {"hate": 0.91, "sexual/minors": 0.0}

    class _Result:
        flagged = True
        category_scores = _Scores()

    async def _flagged(**_: Any) -> Any:
        class _Response:
            results = [_Result()]

        return _Response()

    _install_stub(monkeypatch, _flagged)
    verdict = await moderation_verdict(Settings(), "something vile")
    assert verdict.flagged is True
    assert verdict.available is True
    assert verdict.scores["hate"] == pytest.approx(0.91)


async def test_verdict_does_not_outlive_the_request_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hanging moderator must surface well inside the 60s public route limit."""

    async def _hang(**_: Any) -> Any:
        await asyncio.sleep(30)
        raise AssertionError("should have been abandoned long before this")

    _install_stub(monkeypatch, _hang)
    verdict = await asyncio.wait_for(
        moderation_verdict(Settings(), "hello"), timeout=MODERATION_TIMEOUT_S + 1
    )
    assert verdict.available is False


async def test_client_is_rebuilt_when_the_key_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rotated key must not keep moderating under the previous credential."""
    monkeypatch.setattr(moderation, "_client", None)
    monkeypatch.setattr(moderation, "_client_fingerprint", None)

    monkeypatch.setenv("OPENAI_API_KEY", "first-key")
    first = await moderation._get_client(Settings())
    assert first is await moderation._get_client(Settings())

    monkeypatch.setenv("OPENAI_API_KEY", "second-key")
    second = await moderation._get_client(Settings())
    assert second is not first
    assert first is not None and first.is_closed()
