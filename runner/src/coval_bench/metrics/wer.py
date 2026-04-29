# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Word Error Rate computation.

Uses the legacy text normalization pipeline (currency, ordinals, dates, time
expressions) ported verbatim from ``legacy-benchmarks/wer_calculator.py``.
The DP edit-distance is delegated to ``jiwer`` for correctness and speed.
The normalization pipeline is kept as the default (not ``whisper_normalizer``)
per Phase-2 spec; Phase 4 may A/B test the two pipelines after measuring drift.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Callable
from typing import Literal

import jiwer
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


class WordError(BaseModel):
    type: Literal["substitution", "insertion", "deletion"]
    reference: str | None
    hypothesis: str | None


class WERResult(BaseModel):
    wer: float
    wer_percentage: float
    incorrect_words: list[WordError]
    normalized_reference: str
    normalized_hypothesis: str


# ---------------------------------------------------------------------------
# Private normalization helpers — ported verbatim from legacy wer_calculator.py
# ---------------------------------------------------------------------------


def _dehyphenate(text: str) -> str:
    """Replace hyphens between alphanumeric characters with spaces."""
    words = text.split()
    result: list[str] = []
    for word in words:
        dehyphenated = ""
        for i, ch in enumerate(word):
            if (
                0 < i < len(word) - 1
                and ch == "-"
                and word[i - 1].isalnum()
                and word[i + 1].isalnum()
            ):
                dehyphenated += " "
            else:
                dehyphenated += ch
        result.append(dehyphenated)
    return " ".join(result)


_WORD_TO_NUM: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
}

_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def _sentence_to_numbers(sentence: str) -> str:
    """Convert written numbers to digit form; handles compound and year patterns."""
    words = sentence.split()
    result_words: list[str] = []
    i = 0
    while i < len(words):
        current_word = words[i]
        current = current_word.lower().rstrip(",.;:!?")

        # Determine date context
        is_date_context = False
        if (
            i > 0
            and words[i - 1].lower().rstrip(",.;:!?") in _MONTHS
            or (
                i > 1
                and words[i - 2].lower().rstrip(",.;:!?") in _MONTHS
                and words[i - 1].lower() == "the"
            )
        ):
            is_date_context = True

        # Year patterns: "twenty twenty-four" → "2024"
        if current in _WORD_TO_NUM and i + 1 < len(words):
            first_num = int(_WORD_TO_NUM[current])
            next_word = words[i + 1].lower().rstrip(",.;:!?")
            is_year_pattern = False
            year_value = ""

            if first_num >= 20 and first_num % 10 == 0 and first_num < 100:
                if "-" in next_word:
                    parts = next_word.split("-")
                    if (
                        len(parts) == 2
                        and all(p in _WORD_TO_NUM for p in parts)
                        and _WORD_TO_NUM[parts[0]].endswith("0")
                        and int(_WORD_TO_NUM[parts[1]]) < 10
                    ):
                        second_num = int(_WORD_TO_NUM[parts[0]]) + int(_WORD_TO_NUM[parts[1]])
                        if second_num < 100:
                            year_value = f"{first_num}{second_num:02d}"
                            is_year_pattern = True
                elif next_word in _WORD_TO_NUM:
                    second_num = int(_WORD_TO_NUM[next_word])
                    if second_num >= 20 and second_num % 10 == 0:
                        year_value = f"{first_num}{second_num // 10}0"
                        is_year_pattern = True
                    elif second_num < 20:
                        year_value = f"{first_num}{second_num:02d}"
                        is_year_pattern = True

            if (is_date_context or is_year_pattern) and year_value:
                punctuation = "".join(c for c in words[i + 1] if c in ".,:;!?")
                result_words.append(year_value + punctuation)
                i += 2
                continue

        # Hyphenated number: "twenty-four"
        if "-" in current and any(p in _WORD_TO_NUM for p in current.split("-")):
            parts = current.split("-")
            if all(p in _WORD_TO_NUM for p in parts):
                if _WORD_TO_NUM[parts[0]].endswith("0") and int(_WORD_TO_NUM[parts[1]]) < 10:
                    num_value = int(_WORD_TO_NUM[parts[0]]) + int(_WORD_TO_NUM[parts[1]])
                    punctuation = "".join(c for c in current_word if c in ".,:;!?")
                    result_words.append(str(num_value) + punctuation)
                else:
                    result_words.append(current_word)
            else:
                result_words.append(current_word)

        # Consecutive number words: "twenty four"
        elif current in _WORD_TO_NUM:
            num_value = int(_WORD_TO_NUM[current])
            j = i + 1
            compound_found = False
            while j < len(words) and words[j].lower().rstrip(",.;:!?") in _WORD_TO_NUM:
                next_num = int(_WORD_TO_NUM[words[j].lower().rstrip(",.;:!?")])
                if num_value > 0 and next_num in {100, 1000, 1_000_000, 1_000_000_000}:
                    num_value *= next_num
                    compound_found = True
                elif num_value % 10 == 0 and next_num < 10:
                    num_value += next_num
                    compound_found = True
                else:
                    break
                j += 1
            if compound_found:
                punctuation = "".join(c for c in words[j - 1] if c in ".,:;!?")
                result_words.append(str(num_value) + punctuation)
                i = j - 1
            else:
                punctuation = "".join(c for c in current_word if c in ".,:;!?")
                result_words.append(str(num_value) + punctuation)
        else:
            result_words.append(current_word)

        i += 1

    result = " ".join(result_words)
    result = _convert_ordinals(result)
    result = _clean_time_expressions(result)
    return result


_ORDINAL_MAP: dict[str, str] = {
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "fourth": "4th",
    "fifth": "5th",
    "sixth": "6th",
    "seventh": "7th",
    "eighth": "8th",
    "ninth": "9th",
    "tenth": "10th",
    "eleventh": "11th",
    "twelfth": "12th",
    "thirteenth": "13th",
    "fourteenth": "14th",
    "fifteenth": "15th",
    "sixteenth": "16th",
    "seventeenth": "17th",
    "eighteenth": "18th",
    "nineteenth": "19th",
    "twentieth": "20th",
    "thirtieth": "30th",
    "fortieth": "40th",
    "fiftieth": "50th",
    "sixtieth": "60th",
    "seventieth": "70th",
    "eightieth": "80th",
    "ninetieth": "90th",
    "hundredth": "100th",
    "thousandth": "1000th",
    "millionth": "1000000th",
    "billionth": "1000000000th",
}

_ORDINAL_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_ORDINAL_UNITS: dict[str, tuple[int, str]] = {
    "first": (1, "st"),
    "second": (2, "nd"),
    "third": (3, "rd"),
    "fourth": (4, "th"),
    "fifth": (5, "th"),
    "sixth": (6, "th"),
    "seventh": (7, "th"),
    "eighth": (8, "th"),
    "ninth": (9, "th"),
}


def _convert_ordinals(sentence: str) -> str:
    """Convert ordinal words (e.g. 'first', 'twenty-first') to digit+suffix form."""
    words = sentence.split()
    result_words: list[str] = []
    for current_word in words:
        current = current_word.lower().rstrip(",.;:!?")
        punctuation = "".join(c for c in current_word if c in ".,:;!?")

        if "-" in current:
            parts = current.split("-")
            if len(parts) == 2 and parts[0] in _ORDINAL_TENS and parts[1] in _ORDINAL_UNITS:
                num_value = _ORDINAL_TENS[parts[0]] + _ORDINAL_UNITS[parts[1]][0]
                suffix = _ORDINAL_UNITS[parts[1]][1]
                if num_value % 100 in {11, 12, 13}:
                    suffix = "th"
                result_words.append(f"{num_value}{suffix}{punctuation}")
                continue

        if current in _ORDINAL_MAP:
            mapped = _ORDINAL_MAP[current]
            if current_word[0].isupper():
                result_words.append(mapped.capitalize() + punctuation)
            else:
                result_words.append(mapped + punctuation)
        else:
            result_words.append(current_word)

    return " ".join(result_words)


def _clean_time_expressions(sentence: str) -> str:
    """Normalize time expressions: '3 : 30 PM' → '3:30PM'."""
    sentence = re.sub(r"(\d+)(\s+)([aApP][mM])", r"\1\3", sentence)
    sentence = re.sub(r"(\d+)(\s+)(o'clock|o'Clock|O'clock|O'Clock)", r"\1\3", sentence)
    sentence = re.sub(r"(\d+)(\s*):(\s*)(\d+)(\s+)([aApP][mM])", r"\1:\4\6", sentence)
    sentence = re.sub(r"(\d+)(\s*):(\s*)(\d+)", r"\1:\4", sentence)
    return sentence


_CURRENCY_WORDS: dict[str, str] = {
    "$": "dollars",
    "€": "euros",
    "£": "pounds",
    "¥": "yen",
    "₹": "rupees",
    "₽": "rubles",
    "₩": "won",
    "₿": "bitcoin",
    "₺": "lira",
    "₴": "hryvnia",
    "₼": "manat",
    "₾": "lari",
    "฿": "baht",
    "₫": "dong",
    "₱": "pesos",
    "₦": "naira",
}

_CURRENCY_CODES: dict[str, str] = {
    "USD": "dollars",
    "EUR": "euros",
    "GBP": "pounds",
    "JPY": "yen",
    "INR": "rupees",
    "RUB": "rubles",
    "KRW": "won",
    "BTC": "bitcoin",
    "TRY": "lira",
    "UAH": "hryvnia",
    "AZN": "manat",
    "GEL": "lari",
    "THB": "baht",
    "VND": "dong",
    "PHP": "pesos",
    "NGN": "naira",
}

_CURRENCY_SYMBOL_CHARS = "".join(_CURRENCY_WORDS.keys())


def _format_money(sentence: str) -> str:
    """Convert currency symbols/codes + amounts to word form."""

    def _replace_leading(m: re.Match[str]) -> str:
        symbol = m.group(1)
        whole = m.group(2)
        decimal = m.group(3)
        word = _CURRENCY_WORDS.get(symbol, symbol)
        if decimal:
            return f"{whole} {word} and {decimal} cents"
        return f"{whole} {word}"

    def _replace_trailing(m: re.Match[str]) -> str:
        whole = m.group(1)
        decimal = m.group(2)
        symbol = m.group(3)
        word = _CURRENCY_WORDS.get(symbol, symbol)
        if decimal:
            return f"{whole} {word} and {decimal} cents"
        return f"{whole} {word}"

    def _replace_code(m: re.Match[str]) -> str:
        whole = m.group(1)
        decimal = m.group(2)
        code = m.group(3)
        word = _CURRENCY_CODES.get(code, code.lower())
        if decimal:
            return f"{whole} {word} and {decimal} cents"
        return f"{whole} {word}"

    escaped = re.escape(_CURRENCY_SYMBOL_CHARS)
    result = re.sub(
        rf"([{escaped}])([0-9,]+)(?:\.([0-9]+))?",
        _replace_leading,
        sentence,
    )
    result = re.sub(
        rf"([0-9,]+)(?:\.([0-9]+))?([{escaped}])",
        _replace_trailing,
        result,
    )
    codes = "|".join(_CURRENCY_CODES.keys())
    result = re.sub(
        rf"([0-9,]+)(?:\.([0-9]+))?\s*({codes})",
        _replace_code,
        result,
    )
    return result


def _remove_punctuation(text: str) -> str:
    """Remove punctuation, preserving digit groups (decimal points split to spaces)."""
    text = re.sub(r"(\d+)\.(\d+)", r"\1 \2", text)
    translator = str.maketrans("", "", string.punctuation)
    cleaned = text.translate(translator)
    return " ".join(cleaned.split())


def _squish_numbers(sentence: str) -> str:
    """Remove spaces between consecutive digit characters."""
    chars = list(sentence)
    if len(chars) < 3:
        return sentence
    filtered: list[str] = []
    for i, ch in enumerate(chars):
        if i == 0 or i == len(chars) - 1:
            filtered.append(ch)
            continue
        if chars[i - 1].isdigit() and ch == " " and chars[i + 1].isdigit():
            continue
        filtered.append(ch)
    return "".join(filtered)


# ---------------------------------------------------------------------------
# Public normalization entrypoint
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Canonical normalization pipeline: lower → dehyphenate → numbers → ordinals →
    format_money → remove_punctuation → squish_numbers.

    NFKC Unicode normalization is applied first to handle smart-quotes and ligatures.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _dehyphenate(text)
    text = _sentence_to_numbers(text)
    # Note: _sentence_to_numbers already calls _convert_ordinals and
    # _clean_time_expressions internally (matching legacy behaviour).
    text = _format_money(text)
    text = _remove_punctuation(text)
    text = _squish_numbers(text)
    return text


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------


def _build_word_errors(ref_words: list[str], hyp_words: list[str]) -> list[WordError]:
    """Build the incorrect-words list via a hand-rolled DP backtrace.

    Mirrors the legacy ``find_incorrect_words`` implementation so that the
    error breakdown is consistent with historical data.
    """
    n = len(ref_words)
    m = len(hyp_words)
    # Build edit-distance matrix
    d: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for ii in range(n + 1):
        d[ii][0] = ii
    for jj in range(m + 1):
        d[0][jj] = jj
    for ii in range(1, n + 1):
        for jj in range(1, m + 1):
            if ref_words[ii - 1] == hyp_words[jj - 1]:
                d[ii][jj] = d[ii - 1][jj - 1]
            else:
                d[ii][jj] = 1 + min(
                    d[ii - 1][jj - 1],  # substitution
                    d[ii][jj - 1],  # insertion
                    d[ii - 1][jj],  # deletion
                )

    # Backtrace
    errors: list[WordError] = []
    ii, jj = n, m
    while ii > 0 or jj > 0:
        if ii > 0 and jj > 0 and ref_words[ii - 1] == hyp_words[jj - 1]:
            ii -= 1
            jj -= 1
        elif ii > 0 and jj > 0 and d[ii][jj] == d[ii - 1][jj - 1] + 1:
            errors.append(
                WordError(
                    type="substitution",
                    reference=ref_words[ii - 1],
                    hypothesis=hyp_words[jj - 1],
                )
            )
            ii -= 1
            jj -= 1
        elif jj > 0 and d[ii][jj] == d[ii][jj - 1] + 1:
            errors.append(WordError(type="insertion", reference=None, hypothesis=hyp_words[jj - 1]))
            jj -= 1
        elif ii > 0 and d[ii][jj] == d[ii - 1][jj] + 1:
            errors.append(WordError(type="deletion", reference=ref_words[ii - 1], hypothesis=None))
            ii -= 1
        else:
            # Safety: shouldn't happen, but avoid infinite loop
            if ii > 0:
                ii -= 1
            if jj > 0:
                jj -= 1

    errors.reverse()
    return errors


def compute_wer(
    reference: str,
    hypothesis: str,
    *,
    normalizer: Callable[[str], str] | None = None,
) -> WERResult:
    """Compute Word Error Rate between *reference* and *hypothesis*.

    Args:
        reference: Ground-truth transcript.
        hypothesis: Predicted transcript.
        normalizer: Optional callable that replaces the default
            :func:`normalize_text` pipeline.

    Returns:
        :class:`WERResult` with ``wer`` in [0, ∞), ``wer_percentage``,
        ``incorrect_words``, and both normalized strings.

    Notes:
        - Empty reference + empty hypothesis → ``wer == 0.0``.
        - Empty reference + non-empty hypothesis → ``wer`` is large (jiwer
          returns ``inf``; we cap at the hypothesis word count to keep the
          value finite and consistently comparable).
        - ``wer`` is a ratio, not a percentage; multiply by 100 for ``%``.
    """
    _norm = normalizer if normalizer is not None else normalize_text

    norm_ref = _norm(reference)
    norm_hyp = _norm(hypothesis)

    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()

    # Both empty → perfect match
    if not ref_words and not hyp_words:
        return WERResult(
            wer=0.0,
            wer_percentage=0.0,
            incorrect_words=[],
            normalized_reference=norm_ref,
            normalized_hypothesis=norm_hyp,
        )

    # Empty reference with non-empty hypothesis: jiwer would return inf.
    # We return the insertion count / max(1, hyp_len) so callers always get
    # a finite float. (This matches the convention documented in test_wer.py.)
    if not ref_words:
        ins_count = len(hyp_words)
        empty_ref_wer = float(ins_count)
        errors = [WordError(type="insertion", reference=None, hypothesis=w) for w in hyp_words]
        return WERResult(
            wer=empty_ref_wer,
            wer_percentage=empty_ref_wer * 100.0,
            incorrect_words=errors,
            normalized_reference=norm_ref,
            normalized_hypothesis=norm_hyp,
        )

    # Use jiwer for the WER ratio (fast, correctness-tested Levenshtein)
    raw_wer: float = jiwer.wer(norm_ref, norm_hyp)

    incorrect = _build_word_errors(ref_words, hyp_words)

    return WERResult(
        wer=raw_wer,
        wer_percentage=raw_wer * 100.0,
        incorrect_words=incorrect,
        normalized_reference=norm_ref,
        normalized_hypothesis=norm_hyp,
    )
