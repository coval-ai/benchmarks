# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for partner-scoped early access.

A scope names the embargoed providers one caller may see. Everything else under
embargo stays hidden, so a grant for one vendor never reveals another's
unreleased numbers — and a provider embargoed later is hidden from existing
grants without anyone editing them.

Assertions derive the embargoed set from the registry rather than naming
providers, so they keep holding as models change status.
"""

from __future__ import annotations

import pytest

from coval_bench.api.internal import (
    UNRESTRICTED,
    early_access_scope,
    embargoed_providers,
    hidden_models,
    hidden_models_for,
)
from coval_bench.config import Settings
from coval_bench.registries import MODEL_REGISTRY, Benchmark, ModelStatus

_BFF_KEY = "bff-key-for-tests"


def _embargoed_pairs() -> frozenset[tuple[str, str]]:
    return frozenset(
        (m.provider, m.model) for m in MODEL_REGISTRY if m.status is ModelStatus.EARLY_ACCESS
    )


def _one_embargoed_provider() -> str:
    """Any embargoed provider, or skip.

    At launch every model may be ACTIVE, and these cases are about the mechanism
    rather than the roster — so an empty registry skips instead of erroring.
    """
    providers = sorted(embargoed_providers())
    if not providers:
        pytest.skip("no embargoed providers in the registry")
    return providers[0]


def _embargoed_in(benchmark: Benchmark) -> frozenset[str]:
    return frozenset(
        m.provider
        for m in MODEL_REGISTRY
        if m.status is ModelStatus.EARLY_ACCESS and m.benchmark is benchmark
    )


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings with the BFF key configured and no internal key."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "stt-v1")
    monkeypatch.setenv("RUNNER_SHA", "test-sha")
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    monkeypatch.setenv("EARLY_ACCESS_BFF_KEY", _BFF_KEY)
    return Settings()


def test_empty_scope_hides_every_embargoed_model() -> None:
    """The public view: an anonymous caller sees none of them."""
    assert hidden_models_for(frozenset()) == _embargoed_pairs()
    assert hidden_models() == _embargoed_pairs()


def test_unrestricted_scope_hides_nothing() -> None:
    assert hidden_models_for(UNRESTRICTED) == frozenset()


@pytest.mark.parametrize("provider", sorted(embargoed_providers()))
def test_single_provider_scope_reveals_only_that_provider(provider: str) -> None:
    """A narrow grant must stay narrow.

    Without this, a bug that reveals everything whenever *any* scope is present
    would still pass the empty-scope and unrestricted cases.
    """
    hidden = hidden_models_for(frozenset({provider}))

    revealed = _embargoed_pairs() - hidden
    assert revealed, f"{provider} should have at least one embargoed model revealed"
    assert {p for p, _ in revealed} == {provider}
    assert all(p != provider for p, _ in hidden)


def test_s2s_scope_reveals_all_s2s_and_no_stt_or_tts() -> None:
    """The partner case: every embargoed S2S model, nothing else under embargo."""
    scope = _embargoed_in(Benchmark.S2S)
    if not scope:
        pytest.skip("no embargoed S2S providers in the registry")

    hidden = hidden_models_for(scope)
    hidden_benchmarks = {
        m.benchmark
        for m in MODEL_REGISTRY
        if (m.provider, m.model) in hidden and m.status is ModelStatus.EARLY_ACCESS
    }

    assert Benchmark.S2S not in hidden_benchmarks
    for m in MODEL_REGISTRY:
        if m.status is ModelStatus.EARLY_ACCESS and m.benchmark is not Benchmark.S2S:
            assert (m.provider, m.model) in hidden


def test_scope_needs_the_bff_key(settings: Settings) -> None:
    """A scope header alone grants nothing."""
    named = ",".join(sorted(embargoed_providers()))

    assert (
        early_access_scope(internal=False, x_ea_key=None, x_ea_scope=named, settings=settings)
        == frozenset()
    )
    assert (
        early_access_scope(internal=False, x_ea_key="wrong", x_ea_scope=named, settings=settings)
        == frozenset()
    )


def test_scope_honoured_with_the_bff_key(settings: Settings) -> None:
    provider = _one_embargoed_provider()

    assert early_access_scope(
        internal=False, x_ea_key=_BFF_KEY, x_ea_scope=provider, settings=settings
    ) == frozenset({provider})


def test_unknown_scope_names_are_dropped(settings: Settings) -> None:
    """A malformed header is indistinguishable from an absent one."""
    assert (
        early_access_scope(
            internal=False, x_ea_key=_BFF_KEY, x_ea_scope="not-a-provider,,", settings=settings
        )
        == frozenset()
    )


@pytest.mark.parametrize("widening", ["*", "\x00all", "ALL", "%2A"])
def test_scope_cannot_widen_itself_to_unrestricted(settings: Settings, widening: str) -> None:
    """No request value reaches full visibility, even with a valid key."""
    scope = early_access_scope(
        internal=False, x_ea_key=_BFF_KEY, x_ea_scope=widening, settings=settings
    )

    assert scope == frozenset()
    assert hidden_models_for(scope) == _embargoed_pairs()


def test_internal_key_outranks_any_scope(settings: Settings) -> None:
    assert (
        early_access_scope(internal=True, x_ea_key=None, x_ea_scope=None, settings=settings)
        == UNRESTRICTED
    )


def test_no_configured_bff_key_honours_no_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset means no scope is honoured, whatever the request presents."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "stt-v1")
    monkeypatch.setenv("RUNNER_SHA", "test-sha")
    monkeypatch.delenv("EARLY_ACCESS_BFF_KEY", raising=False)
    unconfigured = Settings()

    provider = _one_embargoed_provider()
    assert (
        early_access_scope(
            internal=False, x_ea_key=None, x_ea_scope=provider, settings=unconfigured
        )
        == frozenset()
    )
