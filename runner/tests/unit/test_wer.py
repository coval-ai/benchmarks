# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Golden-case tests for WER computation and the normalization pipeline."""

from __future__ import annotations

import pytest

from coval_bench.metrics.wer import (
    NORM_VERSION,
    WERResult,
    WordError,
    compute_wer,
    normalize_text,
)

# ---------------------------------------------------------------------------
# 20 golden WER cases
# Each entry: (id, ref, hyp, expected_wer_ratio, note)
# ---------------------------------------------------------------------------

_GOLDEN: list[tuple[int, str, str, float, str]] = [
    (1, "hello world", "hello world", 0.0, "identity"),
    (2, "", "", 0.0, "both empty"),
    (3, "hello", "", 1.0, "full deletion"),
    (4, "", "hello", 1.0, "empty ref — insertion count returned as finite WER"),
    (5, "hello world", "hello", 0.5, "one deletion"),
    (6, "hello world", "hello there", 0.5, "one substitution"),
    (7, "For orders over £500", "for orders over 500 pounds", 0.0, "currency + case"),
    (8, "twenty-four hours", "24 hours", 0.0, "hyphenated number"),
    (
        9,
        "December the fifteenth twenty twenty four",
        "december 15th 2024",
        0.25,
        "spoken date — 'the' deletion is the only error",
    ),
    (10, "don't worry", "do not worry", 0.0, "contraction expansion"),
    (11, "3:30 PM", "3 30 PM", 0.0, "clock time"),
    (12, "$347.89", "347 dollars and 89 cents", 0.0, "currency with cents"),
    (13, "the first second and third", "the 1st 2nd and 3rd", 0.0, "ordinals"),
    (14, "abc-def", "abc def", 0.0, "dehyphenation"),
    (15, "NUMBERED THIRTY SIX MEMBERS", "numbered 36 members", 0.0, "compound number"),
    (16, "hello world", "world hello", 1.0, "two substitutions"),
    (17, "a b c d", "a x c d", 0.25, "one substitution / 4-word ref"),
    (18, "a b c", "a b c d", 1 / 3, "one insertion"),
    (19, "a b c d e", "a b c", 0.4, "two deletions"),
    (20, "twenty-first of May", "21st of May", 0.0, "compound ordinal"),
]


@pytest.mark.parametrize(
    "case_id, ref, hyp, expected_wer, note",
    [(*row,) for row in _GOLDEN],
    ids=[f"case-{row[0]}" for row in _GOLDEN],
)
def test_golden_wer(case_id: int, ref: str, hyp: str, expected_wer: float, note: str) -> None:
    result = compute_wer(ref, hyp)
    assert isinstance(result, WERResult), f"case {case_id}: expected WERResult"
    assert result.wer == pytest.approx(expected_wer, rel=1e-6, abs=1e-9), (
        f"case {case_id} ({note}): got wer={result.wer!r}, expected {expected_wer!r}. "
        f"norm_ref={result.normalized_reference!r}, "
        f"norm_hyp={result.normalized_hypothesis!r}"
    )
    assert result.wer_percentage == pytest.approx(expected_wer * 100, rel=1e-6, abs=1e-9), (
        f"case {case_id}: wer_percentage mismatch"
    )


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

_NORMALIZATION: list[tuple[str, str]] = [
    ("thirty six", "36"),
    ("one hundred twenty three", "123"),
    ("twenty twenty four", "2024"),
    ("nineteen eighty four", "1984"),
    ("fifty five thousand", "55000"),
    ("january twenty five", "january 25"),
    ("june twenty-second", "june 22nd"),
    ("3.5 million", "3500000"),
    ("three point five million", "3500000"),
    ("in 1969, 400 people left", "in 1969 400 people left"),
    ("50%", "50%"),
    ("fifty percent", "50%"),
    ("$3.50", "$3.50"),
    ("three dollars and fifty cents", "$3.50"),
    ("24-48 hours", "24 48 hours"),
    ("version 12.4.1", "version 12.4 one"),
    ("two among them", "2 among them"),
    ("twenty-four", "24"),
    ("don't worry", "do not worry"),
    ("um hello", "hello"),
    ("colour", "color"),
    ("“hello”", "hello"),
    ("naïve café", "naive cafe"),
    ("", ""),
]


@pytest.mark.parametrize(("text", "expected"), _NORMALIZATION, ids=[t for t, _ in _NORMALIZATION])
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


_KNOWN_GAPS: list[tuple[str, str]] = [
    ("two fifteen pm", "215 pm"),
    ("2:15 PM", "2 15 pm"),
    ("a hundred", "a 100"),
    ("100 USD", "100 usd"),
]


@pytest.mark.parametrize(("text", "expected"), _KNOWN_GAPS, ids=[t for t, _ in _KNOWN_GAPS])
def test_normalize_text_known_gaps(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


def test_normalize_text_identical_round_trip() -> None:
    assert normalize_text("hello world") == normalize_text("hello world")


# ---------------------------------------------------------------------------
# Spec acceptance-criteria spot checks
# ---------------------------------------------------------------------------


def test_identity_wer_zero() -> None:
    assert compute_wer("hello world", "hello world").wer == 0.0


def test_wer_percentage_approx() -> None:
    result = compute_wer("a b c", "a x c")
    assert result.wer_percentage == pytest.approx(33.333, abs=0.01)


def test_norm_version() -> None:
    assert NORM_VERSION == "2"
    assert compute_wer("hello", "hello").norm_version == "2"


# ---------------------------------------------------------------------------
# incorrect_words shape
# ---------------------------------------------------------------------------


def test_incorrect_words_substitution() -> None:
    result = compute_wer("hello world", "hello there")
    assert len(result.incorrect_words) == 1
    err = result.incorrect_words[0]
    assert err.type == "substitution"
    assert err.reference == "world"
    assert err.hypothesis == "there"


def test_incorrect_words_deletion() -> None:
    result = compute_wer("a b c", "a c")
    assert any(e.type == "deletion" for e in result.incorrect_words)


def test_incorrect_words_insertion() -> None:
    result = compute_wer("a b", "a b c")
    assert any(e.type == "insertion" for e in result.incorrect_words)


def test_incorrect_words_empty_on_match() -> None:
    result = compute_wer("hello", "hello")
    assert result.incorrect_words == []


# ---------------------------------------------------------------------------
# WordError model
# ---------------------------------------------------------------------------


def test_word_error_model() -> None:
    w = WordError(type="substitution", reference="foo", hypothesis="bar")
    assert w.type == "substitution"
    assert w.reference == "foo"
    assert w.hypothesis == "bar"


# ---------------------------------------------------------------------------
# Custom normalizer
# ---------------------------------------------------------------------------


def test_custom_normalizer() -> None:
    result = compute_wer("Hello World", "hello world", normalizer=str.lower)
    assert result.wer == 0.0
