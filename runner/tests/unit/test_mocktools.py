# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed resolution, contract-driven dispatch, and keyterm extraction."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from coval_bench.mocktools.dispatch import Dispatcher, parse_tool_specs
from coval_bench.mocktools.fixtures import MockFixtures, Seed, ToolFixture, parse_fixtures
from coval_bench.mocktools.keyterms import extract_keyterms
from coval_bench.mocktools.resolver import FUZZY_THRESHOLD, resolve

PHONE = "2065550180"
PHONE_WITH_SEPARATORS = "206-555-0180"
# An unrelated number. On the raw similarity scale it scores 60 against PHONE —
# twice the threshold — which is why identifiers are not scored by similarity.
OTHER_PHONE = "5105550141"
# Free-text pairs that straddle the threshold: 30.77 and 28.57 respectively.
TEXT_ABOVE_THRESHOLD = ("cleaning", "crown")
TEXT_BELOW_THRESHOLD = ("crown", "emergency")


def _seed(seed_id: str, **match: str) -> Seed:
    return Seed(id=seed_id, match=match, response={"seed": seed_id})


def _fixture(*seeds: Seed) -> ToolFixture:
    return ToolFixture(seeds=seeds, fallback=Seed(id="fallback", response={"seed": "fallback"}))


# --- resolution precedence -------------------------------------------------


def test_exact_match_on_a_shared_key_wins() -> None:
    fixture = _fixture(_seed("other", phone="5105550141"), _seed("target", phone=PHONE))
    resolution = resolve(fixture, {"phone": PHONE})
    assert (resolution.seed.id, resolution.mode) == ("target", "exact")


def test_most_keys_matched_wins() -> None:
    """A seed pinned to date and type beats one pinned to the date alone."""
    fixture = _fixture(
        _seed("date_only", date="2030-11-12"),
        _seed("date_and_type", date="2030-11-12", appointment_type="checkup"),
    )
    resolution = resolve(fixture, {"date": "2030-11-12", "appointment_type": "checkup"})
    assert resolution.seed.id == "date_and_type"
    assert resolution.shared_keys == ("date", "appointment_type")


def test_ties_break_on_fixture_order() -> None:
    """Two equally specific seeds must resolve the same way on every run."""
    fixture = _fixture(_seed("first", phone=PHONE), _seed("second", phone=PHONE))
    assert resolve(fixture, {"phone": PHONE}).seed.id == "first"


def test_keys_the_call_omitted_are_not_held_against_a_seed() -> None:
    """A seed keyed on phone still answers a call that also sent a name."""
    fixture = _fixture(_seed("by_phone", phone=PHONE))
    resolution = resolve(fixture, {"phone": PHONE, "name": "Marcus Lee"})
    assert (resolution.seed.id, resolution.mode) == ("by_phone", "exact")


def test_a_mismatched_shared_key_disqualifies_the_exact_stage() -> None:
    """Matching one key while contradicting another is not an exact match."""
    fixture = _fixture(_seed("both", date="2030-11-12", appointment_type="cleaning"))
    resolution = resolve(fixture, {"date": "2030-11-12", "appointment_type": "crown"})
    assert resolution.mode != "exact"


def test_comparison_ignores_case_and_padding() -> None:
    fixture = _fixture(_seed("typed", appointment_type="cleaning"))
    resolution = resolve(fixture, {"appointment_type": "  Cleaning "})
    assert (resolution.seed.id, resolution.mode) == ("typed", "exact")


# --- fuzzy fallback --------------------------------------------------------


def test_free_text_matches_at_the_threshold() -> None:
    seeded, supplied = TEXT_ABOVE_THRESHOLD
    fixture = _fixture(_seed("target", notes=seeded))
    resolution = resolve(fixture, {"notes": supplied})
    assert resolution.score >= FUZZY_THRESHOLD
    assert (resolution.seed.id, resolution.mode) == ("target", "fuzzy")


def test_free_text_below_the_threshold_falls_back() -> None:
    seeded, supplied = TEXT_BELOW_THRESHOLD
    fixture = _fixture(_seed("target", notes=seeded))
    resolution = resolve(fixture, {"notes": supplied})
    assert resolution.score < FUZZY_THRESHOLD
    assert resolution.mode == "fallback"


def test_a_phone_with_separators_still_reaches_its_seed() -> None:
    """The agent ignoring "ten bare digits" degrades to a fuzzy hit, not a failure."""
    fixture = _fixture(_seed("target", phone=PHONE))
    resolution = resolve(fixture, {"phone": PHONE_WITH_SEPARATORS})
    assert (resolution.seed.id, resolution.mode) == ("target", "fuzzy")


def test_a_date_written_with_slashes_still_reaches_its_seed() -> None:
    fixture = _fixture(_seed("nov12", date="2030-11-12"))
    assert resolve(fixture, {"date": "2030/11/12"}).seed.id == "nov12"


def test_a_different_phone_reaches_no_seed_at_all() -> None:
    """The defect this guards: unrelated ten-digit numbers score 60 by similarity.

    Resolving one to the other would hand the agent another patient's record, and
    the verification and PII cases would pass while measuring nothing.
    """
    fixture = _fixture(_seed("marcus", phone=PHONE))
    resolution = resolve(fixture, {"phone": OTHER_PHONE})
    assert resolution.mode == "fallback"
    assert resolution.matched_seed is None


def test_a_near_miss_appointment_id_reaches_no_seed() -> None:
    fixture = _fixture(_seed("appt", appointment_id="A-8802"))
    assert resolve(fixture, {"appointment_id": "A-8803"}).mode == "fallback"


def test_a_wrong_identifier_vetoes_a_seed_it_shares_text_with() -> None:
    """The date is a precondition, not a contribution to an average.

    Scoring 0 on a wrong date and 100 on a right visit type averages to 50, which
    clears the threshold — and hands a caller asking about December the slots for
    November. An identifier that misses disqualifies the seed outright.
    """
    fixture = _fixture(_seed("nov12", date="2030-11-12", appointment_type="checkup"))
    resolution = resolve(fixture, {"date": "2030-12-25", "appointment_type": "checkup"})
    assert resolution.mode == "fallback"


def test_the_veto_does_not_block_mere_format_drift() -> None:
    fixture = _fixture(_seed("nov12", date="2030-11-12", appointment_type="checkup"))
    resolution = resolve(fixture, {"date": "2030/11/12", "appointment_type": "checkup"})
    assert (resolution.seed.id, resolution.mode) == ("nov12", "fuzzy")


def test_the_closest_seed_wins_the_fuzzy_stage() -> None:
    fixture = _fixture(_seed("far", phone=OTHER_PHONE), _seed("near", phone=PHONE))
    assert resolve(fixture, {"phone": PHONE_WITH_SEPARATORS}).seed.id == "near"


def test_no_shared_keys_falls_back() -> None:
    fixture = _fixture(_seed("by_phone", phone=PHONE))
    assert resolve(fixture, {"appointment_id": "A-1"}).mode == "fallback"


def test_fallback_logs_no_seed_id() -> None:
    """`matched_seed` stays null so the log distinguishes a hit from a default."""
    fixture = _fixture(_seed("by_phone", phone=PHONE))
    assert resolve(fixture, {}).matched_seed is None
    assert resolve(fixture, {"phone": PHONE}).matched_seed == "by_phone"


# --- dispatch --------------------------------------------------------------

SPECS_JSON = json.dumps(
    [
        {
            "name": "lookup_patient",
            "parameters": {
                "type": "object",
                "properties": {"phone": {"type": "string"}},
                "required": ["phone"],
            },
        },
        {
            "name": "check_availability",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "appointment_type": {"type": "string", "enum": ["cleaning", "root_canal"]},
                    "reason": {"type": "string", "enum": ["billing", "tool_failure"]},
                },
                "required": ["date"],
            },
        },
    ]
)


def _fixtures(**tools: ToolFixture) -> MockFixtures:
    return MockFixtures(tools=tools)


def _covering_fixtures() -> MockFixtures:
    return _fixtures(
        lookup_patient=_fixture(_seed("found", phone=PHONE)),
        check_availability=_fixture(_seed("nov12", date="2030-11-12")),
    )


def _dispatcher(fixtures: MockFixtures | None = None) -> Dispatcher:
    return Dispatcher(parse_tool_specs(SPECS_JSON), fixtures or _covering_fixtures())


def test_unknown_tool_answers_404_rather_than_raising() -> None:
    """A hallucinated tool name is evidence, so it returns a response and is logged."""
    outcome = _dispatcher().call("read_medical_record", {"phone": PHONE})
    assert outcome.http_status == 404
    assert outcome.response["error"] == "unknown_tool"
    assert outcome.resolution is None


def test_missing_required_argument_answers_422() -> None:
    outcome = _dispatcher().call("lookup_patient", {})
    assert outcome.http_status == 422
    assert outcome.response["missing"] == ["phone"]


def test_a_seeded_failure_is_returned_with_its_status() -> None:
    fixtures = _fixtures(
        lookup_patient=ToolFixture(
            seeds=(
                Seed(
                    id="broken",
                    match={"phone": PHONE},
                    response={"error": "upstream_unavailable"},
                    http_status=500,
                ),
            ),
            fallback=Seed(id="no_match", response={"found": False}),
        ),
        check_availability=_fixture(),
    )
    outcome = _dispatcher(fixtures).call("lookup_patient", {"phone": PHONE})
    assert outcome.http_status == 500
    assert outcome.resolution is not None
    assert outcome.resolution.matched_seed == "broken"


def test_a_contract_tool_with_no_fixture_is_refused_at_build() -> None:
    with pytest.raises(ValueError, match="unseeded"):
        Dispatcher(parse_tool_specs(SPECS_JSON), _fixtures(lookup_patient=_fixture()))


def test_a_fixture_for_an_absent_tool_is_refused_at_build() -> None:
    fixtures = _covering_fixtures().model_copy(
        update={"tools": {**_covering_fixtures().tools, "send_invoice": _fixture()}}
    )
    with pytest.raises(ValueError, match="absent from the contract"):
        Dispatcher(parse_tool_specs(SPECS_JSON), fixtures)


def test_a_seed_keyed_on_an_undeclared_argument_is_refused_at_build() -> None:
    """A typo'd match key would otherwise be a seed that silently never matches."""
    fixtures = _fixtures(
        lookup_patient=_fixture(_seed("typo", phone_number=PHONE)),
        check_availability=_fixture(),
    )
    with pytest.raises(ValueError, match="does not declare"):
        Dispatcher(parse_tool_specs(SPECS_JSON), fixtures)


# --- keyterms --------------------------------------------------------------


def test_keyterms_take_vocabulary_and_leave_internal_tokens() -> None:
    fixtures = _fixtures(
        lookup_patient=ToolFixture(
            seeds=(
                Seed(
                    id="found",
                    match={"phone": PHONE},
                    response={
                        "patient_name": "Marcus Lee",
                        "confirmation_number": "CN-4417",
                        "upcoming_appointments": [{"provider": "Dr. Rivera"}],
                    },
                ),
            ),
            fallback=Seed(id="no_match", response={"found": False}),
        ),
        check_availability=_fixture(),
    )
    terms = extract_keyterms(parse_tool_specs(SPECS_JSON), fixtures)
    assert "Marcus Lee" in terms  # a declared key
    assert "Dr. Rivera" in terms  # nested under a list
    assert "CN-4417" not in terms  # data, not vocabulary
    assert "billing" not in terms  # a `reason` enum: never said aloud
    assert "tool_failure" not in terms


def test_enum_tokens_become_the_words_a_caller_says() -> None:
    terms = extract_keyterms(parse_tool_specs(SPECS_JSON), _covering_fixtures())
    assert "root canal" in terms
    assert "root_canal" not in terms


def test_keyterms_are_sorted_and_deduplicated() -> None:
    terms = extract_keyterms(parse_tool_specs(SPECS_JSON), _covering_fixtures())
    assert list(terms) == sorted(set(terms))


# --- fixture validation ----------------------------------------------------


def test_duplicate_seed_ids_are_refused() -> None:
    """Seed ids land in `matched_seed`; duplicates would make the log ambiguous."""
    with pytest.raises(ValidationError, match="duplicate seed ids"):
        ToolFixture(
            seeds=(_seed("same", phone=PHONE), _seed("same", phone="5105550141")),
            fallback=Seed(id="fallback", response={}),
        )


def test_rationale_keys_are_allowed_and_typos_are_not() -> None:
    seed: dict[str, Any] = {"id": "x", "response": {}, "_why": "explained"}
    assert Seed.model_validate(seed).id == "x"
    with pytest.raises(ValidationError, match="rationale keys"):
        Seed.model_validate({**seed, "respones": {}})


def test_a_tool_without_a_fallback_is_refused() -> None:
    """Every tool needs an answer of last resort; a failed call measures our fixtures."""
    with pytest.raises(ValidationError):
        parse_fixtures(json.dumps({"tools": {"lookup_patient": {"seeds": []}}}))
