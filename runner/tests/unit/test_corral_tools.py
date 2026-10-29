# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A Corral ``tools.json`` read as a mock suite: fixed answers, argument checks kept."""

from __future__ import annotations

import json

import pytest

from coval_bench.mocktools.dispatch import (
    CORRAL_SUITES,
    build_corral_dispatcher,
    parse_corral_tools,
)
from coval_bench.scenarios import PUBLIC_CONTRACT_FILES, public_contract_sha256


def _tools(**overrides: object) -> str:
    document = {
        "tools": [
            {
                "name": "lookup",
                "description": "Look it up.",
                "parameters": {"value": "string"},
                "character_error_rate": {"value": {"account-1": "KD7B"}},
            }
        ],
        "responses": {"lookup": {"received": True}},
    }
    return json.dumps({**document, **overrides})


def test_every_corral_suite_loads_from_its_committed_file() -> None:
    for suite in CORRAL_SUITES:
        assert build_corral_dispatcher(suite).tools


def test_the_tools_file_is_part_of_the_contract_hash() -> None:
    assert "tools.json" in PUBLIC_CONTRACT_FILES
    assert public_contract_sha256("id_roulette") != public_contract_sha256("no_such_suite")


def test_id_roulette_answers_lookup_and_checks_its_value() -> None:
    dispatcher = build_corral_dispatcher("id_roulette")
    outcome = dispatcher.call("lookup", {"value": "KD7BO0RWNLMX0O4BYD"})
    assert (outcome.http_status, outcome.response) == (200, {"received": True})
    assert set(dispatcher.checks) == {"lookup"}
    assert dispatcher.checks["lookup"]["value"]["account-1"] == "KD7BO0RWNLMX0O4BYD"  # type: ignore[index]


def test_a_missing_or_mistyped_argument_is_rejected_like_any_other_suite() -> None:
    dispatcher = parse_corral_tools(_tools())
    assert dispatcher.call("lookup", {}).http_status == 422
    assert dispatcher.call("lookup", {"value": 42}).http_status == 422
    assert dispatcher.call("other", {"value": "x"}).http_status == 404


def test_per_test_case_answers_are_refused() -> None:
    with pytest.raises(ValueError, match="per-test-case answers"):
        parse_corral_tools(_tools(test_cases={"account-1": {"lookup": {"received": False}}}))


def test_a_check_on_an_argument_the_tool_does_not_take_is_refused() -> None:
    document = json.loads(_tools())
    document["tools"][0]["character_error_rate"] = {"other": "x"}
    with pytest.raises(ValueError, match="checks arguments it does not take"):
        parse_corral_tools(json.dumps(document))


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        json.dumps({"tools": {"lookup": {}}, "responses": {}}),
        json.dumps({"tools": [{"parameters": {}}], "responses": {}}),
        json.dumps({"tools": [{"name": "lookup"}], "responses": [{"received": True}]}),
        json.dumps({"tools": [], "responses": {}}),
        json.dumps(
            {
                "tools": [
                    {
                        "name": "lookup",
                        "parameters": {"value": "string"},
                        "character_error_rate": {"value": ["KD7B"]},
                    }
                ],
                "responses": {"lookup": {}},
            }
        ),
    ],
    ids=["not-json", "tools-not-a-list", "nameless", "responses-a-list", "no-tools", "bad-check"],
)
def test_a_malformed_file_is_a_value_error_not_a_crash(document: str) -> None:
    with pytest.raises(ValueError):
        parse_corral_tools(document)


def test_two_tools_with_one_name_are_refused() -> None:
    document = json.loads(_tools())
    document["tools"].append(document["tools"][0])
    with pytest.raises(ValueError, match="two tools are named"):
        parse_corral_tools(json.dumps(document))


def test_responses_must_match_the_tools() -> None:
    with pytest.raises(ValueError, match="has no response"):
        parse_corral_tools(_tools(responses={}))
    with pytest.raises(ValueError, match="tools that do not exist"):
        parse_corral_tools(_tools(responses={"lookup": {}, "ghost": {}}))
