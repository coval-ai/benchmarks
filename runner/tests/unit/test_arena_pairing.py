# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for arena model pairing (adaptive + cold-start strategies)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from coval_bench.arena.pairing import (
    active_tts_models,
    gender_for_next_battle,
    roster_for,
    select_pair,
    voice_for,
)
from coval_bench.config import Settings
from coval_bench.db.models import PairingRating
from coval_bench.registries import MODEL_REGISTRY
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import Gender, ModelStatus, RegisteredModel, Voice
from coval_bench.registries.provider_keys import PROVIDER_ENV


def _model(name: str) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.TTS,
        provider=name,
        model=name,
        voice="v",
        status=ModelStatus.ACTIVE,
    )


def _gendered_model(provider: str, model: str) -> RegisteredModel:
    """A roster entry with both gendered voices, so ``roster_for`` keeps it."""
    return RegisteredModel(
        benchmark=Benchmark.TTS,
        provider=provider,
        model=model,
        voice="v",
        voices=(
            Voice(id=f"{model}-f", gender=Gender.FEMALE),
            Voice(id=f"{model}-m", gender=Gender.MALE),
        ),
        status=ModelStatus.ACTIVE,
    )


def _rating(elo: float, ci: float | None) -> PairingRating:
    return PairingRating(rating_elo=elo, ci_half_width=ci)


def _key(m: RegisteredModel) -> tuple[str, str]:
    return (m.provider, m.model)


_MODELS = [_model("a"), _model("b"), _model("c")]


def test_select_pair_returns_two_distinct() -> None:
    first, second = select_pair(_MODELS, {}, rng=random.Random(1))
    assert first in _MODELS
    assert second in _MODELS
    assert first is not second


def test_select_pair_requires_two_models() -> None:
    with pytest.raises(ValueError, match="at least two"):
        select_pair(_MODELS[:1], {})


def test_cold_start_is_deterministic_with_seed() -> None:
    assert select_pair(_MODELS, {}, rng=random.Random(7)) == select_pair(
        _MODELS, {}, rng=random.Random(7)
    )


def test_adaptive_is_deterministic_with_seed() -> None:
    ratings = {
        _key(m): _rating(e, 30.0) for m, e in zip(_MODELS, [1000.0, 1100.0, 1200.0], strict=True)
    }
    assert select_pair(_MODELS, ratings, rng=random.Random(7)) == select_pair(
        _MODELS, ratings, rng=random.Random(7)
    )


def test_close_pair_outscores_blowout() -> None:
    # a~b are near-ties, c is a far outlier; the near-tie should dominate draws.
    ratings = {
        _key(m): _rating(e, 50.0) for m, e in zip(_MODELS, [1000.0, 1005.0, 1400.0], strict=True)
    }
    counts: Counter[frozenset[str]] = Counter(
        frozenset((a.model, b.model))
        for a, b in (select_pair(_MODELS, ratings, rng=random.Random(i)) for i in range(500))
    )
    assert counts[frozenset(("a", "b"))] > counts[frozenset(("a", "c"))]


def test_high_ci_model_oversampled_at_equal_gap() -> None:
    # equal ratings, but c is far more uncertain → it should appear more often.
    ratings = {
        _key(m): _rating(1000.0, ci) for m, ci in zip(_MODELS, [10.0, 10.0, 200.0], strict=True)
    }
    counts: Counter[str] = Counter(
        m.model
        for pair in (select_pair(_MODELS, ratings, rng=random.Random(i)) for i in range(500))
        for m in pair
    )
    assert counts["c"] > counts["a"]


def test_new_model_absent_from_ratings_is_reachable() -> None:
    # c has no rating → treated as max uncertainty, so it must still get sampled.
    ratings = {_key(_MODELS[0]): _rating(1000.0, 20.0), _key(_MODELS[1]): _rating(1010.0, 20.0)}
    seen = {
        m.model
        for pair in (select_pair(_MODELS, ratings, rng=random.Random(i)) for i in range(200))
        for m in pair
    }
    assert "c" in seen


def test_all_converged_falls_back_to_uniform() -> None:
    # every model fully converged (ci=0) → weights sum to 0 → uniform fallback.
    # Same seed must reproduce the cold-start pick, proving the fallback branch ran.
    ratings = {
        _key(m): _rating(e, 0.0) for m, e in zip(_MODELS, [1000.0, 1100.0, 1200.0], strict=True)
    }
    assert select_pair(_MODELS, ratings, rng=random.Random(3)) == select_pair(
        _MODELS, {}, rng=random.Random(3)
    )


def test_new_model_prioritized_even_when_known_models_converged() -> None:
    # a, b fully converged (ci=0); c is absent. The max-CI floor keeps c the only
    # model with weight, so every selected pair must include it.
    ratings = {_key(_MODELS[0]): _rating(1000.0, 0.0), _key(_MODELS[1]): _rating(1000.0, 0.0)}
    pairs = [select_pair(_MODELS, ratings, rng=random.Random(i)) for i in range(50)]
    assert all("c" in (a.model, b.model) for a, b in pairs)


def test_no_active_model_rated_falls_back_to_uniform() -> None:
    # Board holds only a retired model absent from the active pool → no signal →
    # uniform, matching the cold-start pick for the same seed.
    ratings = {("retired", "old"): _rating(1500.0, 40.0)}
    assert select_pair(_MODELS, ratings, rng=random.Random(1)) == select_pair(
        _MODELS, {}, rng=random.Random(1)
    )


def test_select_pair_rejects_non_positive_scale() -> None:
    ratings = {_key(m): _rating(1000.0, 30.0) for m in _MODELS}
    with pytest.raises(ValueError, match="scale must be"):
        select_pair(_MODELS, ratings, scale=0.0)


def test_active_tts_models_are_tts_and_active() -> None:
    roster = active_tts_models(MODEL_REGISTRY)
    assert len(roster) >= 2
    assert all(m.benchmark is Benchmark.TTS and m.status is ModelStatus.ACTIVE for m in roster)


def test_active_tts_models_excludes_arena_disabled() -> None:
    assert all(m.arena_enabled for m in active_tts_models(MODEL_REGISTRY))


def test_provider_env_covers_arena_providers() -> None:
    # Only providers the arena can actually synthesize with (ACTIVE + arena_enabled),
    # matching active_tts_models(MODEL_REGISTRY) and the parity script — not every non-retired one.
    providers = {m.provider for m in active_tts_models(MODEL_REGISTRY)}
    missing = providers - PROVIDER_ENV.keys()
    assert not missing, f"arena providers with no PROVIDER_ENV entry: {sorted(missing)}"


def test_provider_env_names_match_settings_fields() -> None:
    fields = Settings.model_fields
    bad = {env for env in PROVIDER_ENV.values() if env.lower() not in fields}
    assert not bad, f"PROVIDER_ENV names with no Settings field: {sorted(bad)}"


def test_every_arena_model_can_field_both_genders() -> None:
    """Any arena model must field both genders, or it silently sits out half the battles.

    Palabra is the standing exception: its ``voices`` are quality tiers rather
    than speakers, so it has no gendered pool to draw from.
    """
    incomplete = [
        f"{m.provider}/{m.model}"
        for m in active_tts_models(MODEL_REGISTRY)
        if {v.gender for v in m.voices} != {Gender.FEMALE, Gender.MALE} and m.provider != "palabra"
    ]
    assert incomplete == [], f"arena models missing a gendered voice: {incomplete}"


def test_roster_for_keeps_only_models_with_that_gender() -> None:
    for gender in (Gender.FEMALE, Gender.MALE):
        roster = roster_for(MODEL_REGISTRY, gender)
        assert len(roster) >= 2
        assert all(any(v.gender is gender for v in m.voices) for m in roster)
    assert {m.provider for m in roster_for(MODEL_REGISTRY, Gender.FEMALE)}.isdisjoint({"palabra"})


def test_voice_for_returns_the_matching_half() -> None:
    model = RegisteredModel(
        benchmark=Benchmark.TTS,
        provider="p",
        model="m",
        voice="v",
        voices=(Voice(id="她", gender=Gender.FEMALE), Voice(id="他", gender=Gender.MALE)),
        status=ModelStatus.ACTIVE,
    )
    assert voice_for(model, Gender.FEMALE).id == "她"
    assert voice_for(model, Gender.MALE).id == "他"


def test_voice_for_raises_when_the_model_cannot_field_the_gender() -> None:
    model = RegisteredModel(
        benchmark=Benchmark.TTS,
        provider="p",
        model="m",
        voice="v",
        voices=(Voice(id="only-female", gender=Gender.FEMALE),),
        status=ModelStatus.ACTIVE,
    )
    with pytest.raises(ValueError, match="no male voice"):
        voice_for(model, Gender.MALE)


def test_gender_for_next_battle_picks_the_deficit() -> None:
    assert gender_for_next_battle({Gender.FEMALE: 6, Gender.MALE: 5}) is Gender.MALE
    assert gender_for_next_battle({Gender.FEMALE: 5, Gender.MALE: 6}) is Gender.FEMALE
    # A concurrent double-pick overshoots; the deficit rule pulls it back rather
    # than latching, which a parity bit would not do.
    assert gender_for_next_battle({Gender.FEMALE: 7, Gender.MALE: 5}) is Gender.MALE
    # An absent gender counts as zero, so a cold start is not a special case.
    assert gender_for_next_battle({Gender.FEMALE: 3}) is Gender.MALE


def test_gender_for_next_battle_breaks_ties_both_ways() -> None:
    """A fixed tie-break would hand one gender a standing +1 at every cold start."""
    picks = {gender_for_next_battle({}, random.Random(seed)) for seed in range(20)}
    assert picks == {Gender.FEMALE, Gender.MALE}


def test_gender_alternates_over_a_run_of_battles() -> None:
    counts: dict[Gender, int] = {}
    for _ in range(20):
        gender = gender_for_next_battle(counts, random.Random(0))
        counts[gender] = counts.get(gender, 0) + 1
    assert counts[Gender.FEMALE] == counts[Gender.MALE] == 10


class TestBenchedProviders:
    """A provider whose key is dead is dropped from the draw — unless dropping it
    would leave no battle to run at all."""

    def test_benched_providers_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        roster = [
            _gendered_model("alpha", "m-alpha"),
            _gendered_model("beta", "m-beta"),
            _gendered_model("gamma", "m-gamma"),
        ]
        remaining = roster_for(roster, Gender.FEMALE, frozenset({"beta"}))

        assert [m.provider for m in remaining] == ["alpha", "gamma"]

    def test_a_benched_provider_is_never_returned_to_make_up_the_numbers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Too few to pair is the caller's problem to report; pairing a key already known
        # to be dead spends a paid call and fails the voter anyway.
        roster = [_gendered_model("alpha", "m-alpha"), _gendered_model("beta", "m-beta")]
        remaining = roster_for(roster, Gender.FEMALE, frozenset({"beta"}))

        assert [m.provider for m in remaining] == ["alpha"]
