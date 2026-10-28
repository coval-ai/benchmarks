# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The copy of Corral's checks, held to Corral's own test cases."""

from __future__ import annotations

import pytest

from coval_bench.mocktools.codecs import ToolCall
from coval_bench.mocktools.tool_checks import character_error_rate, character_error_rates

ID = "KD7BO0RWNLMX0O4BYD"


def test_character_error_rate_counts_edits_over_the_expected_length() -> None:
    assert character_error_rate(ID, ID) == 0.0
    assert character_error_rate(ID, "KD7B00RWNLMX0O4BYD") == pytest.approx(1 / 18)
    assert character_error_rate(ID, "KD7BO0RW") == pytest.approx(10 / 18)
    assert character_error_rate(ID, "") == 1.0


def test_case_spaces_and_punctuation_the_answer_lacks_do_not_count() -> None:
    assert character_error_rate(ID, "kd7 bo0-rwnl mx0o 4byd") == 0.0


def test_the_first_call_is_scored_and_a_missing_one_scores_one() -> None:
    expected = {"lookup_id": {"id": ID}}
    first, retry = ToolCall("lookup_id", {"id": "KD7B"}), ToolCall("lookup_id", {"id": ID})
    assert character_error_rates(expected, [first, retry]) == {
        "lookup_id.id": pytest.approx(14 / 18)
    }
    assert character_error_rates(expected, []) == {"lookup_id.id": 1.0}
    assert character_error_rates(expected, [ToolCall("lookup_id", {})]) == {"lookup_id.id": 1.0}
    assert character_error_rates({}, [first]) == {}


def test_punctuation_counts_where_the_right_answer_has_it() -> None:
    email = "j.okafor_88@example.com"
    assert character_error_rate(email, "J.Okafor_88@Example.com") == 0.0
    assert character_error_rate(email, "jokafor88@example.com") == pytest.approx(2 / 23)
    assert character_error_rate("O'Neill", "O’Neill") == 0.0
    assert character_error_rate("O'Neill", "ONeill") == pytest.approx(1 / 7)
    assert character_error_rate("4242424242424242", "4242-4242-4242-4242") == 0.0
