# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A Corral ``tools.json`` read as a mock suite: fixed answers, argument checks kept."""

from __future__ import annotations

import json

import pytest

from coval_bench import scenarios
from coval_bench.mocktools.dispatch import SUITES, build_suite, parse_corral_tools
from coval_bench.scenarios import (
    contract_sha256,
    public_contract_sha256,
    read_contract_file,
    register_fixture_provider,
)


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


@pytest.fixture
def nothing_private(monkeypatch: pytest.MonkeyPatch) -> None:
    """No local ``_private/`` and no providers, as in CI before anything is registered."""
    monkeypatch.setattr(scenarios, "_FIXTURE_PROVIDERS", [])
    real = scenarios._read_bytes

    def no_local_private(*parts: str) -> bytes:
        if "_private" in parts:
            raise FileNotFoundError("/".join(parts))
        return real(*parts)

    monkeypatch.setattr(scenarios, "_read_bytes", no_local_private)


@pytest.fixture
def published(nothing_private: None) -> None:
    """The bucket carries each suite's tools.json, as the assets publish it."""

    def bucket(suite: str, name: str) -> bytes:
        if name != "tools.json":
            raise FileNotFoundError(f"{suite}/{name}")
        return _tools().encode()

    register_fixture_provider(bucket)


def test_no_suite_keeps_its_own_copy_of_the_tools() -> None:
    """The assets in benchmark-infra are the one place a suite's tools are written."""
    for suite in SUITES:
        with pytest.raises(FileNotFoundError):
            read_contract_file(suite, "tools.json")


def test_a_suite_is_built_from_the_published_file(published: None) -> None:
    suite = build_suite("id_roulette")
    outcome = suite.dispatcher.call("lookup", {"value": "KD7B"})
    assert (outcome.http_status, outcome.response) == (200, {"received": True})
    assert suite.checks == {"lookup": {"value": {"account-1": "KD7B"}}}


def test_without_the_published_file_the_suite_is_not_built(nothing_private: None) -> None:
    with pytest.raises(FileNotFoundError):
        build_suite("id_roulette")


def test_the_published_file_is_part_of_the_contract_hash(published: None) -> None:
    assert contract_sha256("id_roulette") != public_contract_sha256("id_roulette")


def test_a_missing_or_mistyped_argument_is_rejected_like_any_other_suite() -> None:
    dispatcher = parse_corral_tools(_tools()).dispatcher
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
