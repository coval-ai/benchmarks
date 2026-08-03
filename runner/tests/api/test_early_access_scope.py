# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for partner early-access tokens.

A token resolves server-side to an allowlist of the exact models it may see.
Everything else under embargo stays hidden, so a grant for one vendor never
reveals another's unreleased numbers, and a model embargoed later is hidden from
existing grants without anyone editing them.

Assertions derive the embargoed set from the registry rather than naming models,
so they keep holding as the roster changes status.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
from fastapi import Response

from coval_bench.api.internal import (
    EA_STATUS_HEADER,
    VARY_HEADERS,
    embargoed_pairs,
    hidden_early_access,
    hidden_models,
)
from coval_bench.config import Settings

_TOKEN = "token-for-tests"  # noqa: S105 - fake grant token
_OTHER_TOKEN = "another-token-for-tests"  # noqa: S105 - fake grant token


def _settings(monkeypatch: pytest.MonkeyPatch, tokens: object | None) -> Settings:
    monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "stt-v1")
    monkeypatch.setenv("RUNNER_SHA", "test-sha")
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    if tokens is None:
        monkeypatch.delenv("EARLY_ACCESS_TOKENS", raising=False)
    else:
        raw = tokens if isinstance(tokens, str) else json.dumps(tokens)
        monkeypatch.setenv("EARLY_ACCESS_TOKENS", raw)
    return Settings()


def _resolve(
    settings: Settings, token: str | None, internal: bool = False
) -> tuple[frozenset[tuple[str, str]], str]:
    """Return this caller's hidden set and the status header the response carries.

    Every caller is varied on both proof headers, so that is asserted here rather
    than repeated in each case.
    """
    response = Response()
    hidden = hidden_early_access(
        response=response, internal=internal, x_ea_token=token, settings=settings
    )
    assert response.headers["Vary"] == VARY_HEADERS
    return hidden, response.headers[EA_STATUS_HEADER]


def _cache_control(settings: Settings, token: str | None, internal: bool = False) -> str | None:
    response = Response()
    hidden_early_access(response=response, internal=internal, x_ea_token=token, settings=settings)
    return response.headers.get("Cache-Control")


def _hidden(settings: Settings, token: str | None) -> frozenset[tuple[str, str]]:
    return _resolve(settings, token)[0]


def _some_pair() -> tuple[str, str]:
    """Any embargoed pair, or skip.

    At launch every model may be ACTIVE, and these cases are about the mechanism
    rather than the roster — so an empty set skips instead of erroring.
    """
    pairs = sorted(embargoed_pairs())
    if not pairs:
        pytest.skip("no embargoed models in the registry")
    return pairs[0]


def _sibling_pair() -> tuple[tuple[str, str], tuple[str, str]]:
    """Two embargoed models on the same provider, or skip."""
    counts = Counter(provider for provider, _ in embargoed_pairs())
    for provider, count in sorted(counts.items()):
        if count >= 2:
            models = sorted(m for p, m in embargoed_pairs() if p == provider)
            return (provider, models[0]), (provider, models[1])
    pytest.skip("no provider has two embargoed models")


def _entry(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"


def test_retired_board_keys_stay_embargoed() -> None:
    """A renamed embargoed model keeps hiding artefacts stored under its old key.

    Sample manifests and result rows written before the rename still carry the
    old string, and the embargo matches on what is stored -- so losing the old
    key would publish every recording and row published under that name.
    """
    assert ("xai", "grok-realtime") in embargoed_pairs()
    assert ("xai", "grok-voice-think-fast-1.0") in embargoed_pairs()


def test_no_token_hides_every_embargoed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, {_TOKEN: []})
    hidden, status = _resolve(settings, token=None)
    assert hidden == embargoed_pairs()
    assert status == "absent"
    assert hidden_models() == embargoed_pairs()


def test_internal_caller_hides_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, {_TOKEN: []})
    hidden, status = _resolve(settings, token=None, internal=True)
    assert hidden == frozenset()
    assert status == "internal"


def test_token_reveals_exactly_its_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = _some_pair()
    settings = _settings(monkeypatch, {_TOKEN: [_entry(pair)]})
    hidden, status = _resolve(settings, token=_TOKEN)
    assert status == "accepted"
    assert hidden == embargoed_pairs() - {pair}


def test_one_token_never_reveals_anothers_models(monkeypatch: pytest.MonkeyPatch) -> None:
    mine, theirs = _sibling_pair()
    settings = _settings(monkeypatch, {_TOKEN: [_entry(mine)], _OTHER_TOKEN: [_entry(theirs)]})
    hidden = _hidden(settings, token=_TOKEN)
    assert mine not in hidden
    assert theirs in hidden


def test_allowlist_is_per_model_not_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two embargoed models on one provider: naming one must not reveal the other."""
    first, second = _sibling_pair()
    settings = _settings(monkeypatch, {_TOKEN: [_entry(first)]})
    hidden = _hidden(settings, token=_TOKEN)
    assert first not in hidden
    assert second in hidden


def test_unknown_token_gets_the_public_view(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, {_TOKEN: [_entry(_some_pair())]})
    hidden, status = _resolve(settings, token="not-a-configured-token")  # noqa: S106 - fake token
    assert hidden == embargoed_pairs()
    assert status == "unknown"


def test_configured_token_with_an_empty_list_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, {_TOKEN: []})
    hidden, status = _resolve(settings, token=_TOKEN)
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_no_configured_tokens_honours_none(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, None)
    assert _hidden(settings, token=_TOKEN) == embargoed_pairs()


def test_entry_naming_a_model_outside_the_embargo_is_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, {_TOKEN: ["nobody/not-a-real-model"]})
    assert _hidden(settings, token=_TOKEN) == embargoed_pairs()


@pytest.mark.parametrize(
    "blob",
    [
        "not json at all",
        '["a", "list", "not", "an", "object"]',
        '{"token": "a string, not a list"}',
        '{"token": ["missing-the-slash"]}',
        '{"token": ["/no-provider"]}',
        '{"token": ["no-model/"]}',
        '{"token": [42]}',
    ],
)
def test_malformed_config_falls_back_to_the_public_view(
    monkeypatch: pytest.MonkeyPatch, blob: str
) -> None:
    settings = _settings(monkeypatch, blob)
    assert _hidden(settings, token="token") == embargoed_pairs()  # noqa: S106 - fake token


def test_a_model_id_containing_a_slash_still_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _some_pair()
    settings = _settings(monkeypatch, {_TOKEN: [f"{provider}/org/model"]})
    # Split on the first slash, so the model is "org/model" — no such model is
    # embargoed, so nothing is revealed, but the entry parsed rather than raised.
    assert _hidden(settings, token=_TOKEN) == embargoed_pairs()


def test_cache_control_is_private_only_when_a_proof_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A privileged response must never be stored; a public one is cacheable."""
    settings = _settings(monkeypatch, {_TOKEN: [_entry(_some_pair())]})
    assert _cache_control(settings, token=None, internal=True) == "private, no-store"
    assert _cache_control(settings, token=_TOKEN) == "private, no-store"
    assert _cache_control(settings, token=None) is None
    assert _cache_control(settings, token="wrong") is None  # noqa: S106 - fake token
