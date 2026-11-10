# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Word Error Rate computation.

Text normalization is delegated to ``whisper_normalizer``'s
``EnglishTextNormalizer``, the de facto standard for published WER; the DP
edit-distance is delegated to ``jiwer`` for correctness and speed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

import jiwer
from pydantic import BaseModel
from whisper_normalizer.english import EnglishTextNormalizer

# Bumped on any behavioural change to normalization; surfaced on ``WERResult``
# so a result row can be attributed to the revision that produced it.
# Documented in ``docs/methodology.md`` and ADR-021.
NORM_VERSION: Literal["3"] = "3"

# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


class WERResult(BaseModel):
    wer: float
    wer_percentage: float
    normalized_reference: str
    normalized_hypothesis: str
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    norm_version: Literal["3"] = NORM_VERSION

    @property
    def error_counts(self) -> dict[str, int]:
        return {
            "wer_substitutions": self.substitutions,
            "wer_deletions": self.deletions,
            "wer_insertions": self.insertions,
            "wer_reference_words": self.reference_words,
        }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_normalizer = EnglishTextNormalizer()

_ELLIPSIS = re.compile(r"…|\.(?:\s*\.)+")
_INITIALISM = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.)+[A-Za-z]?(?![A-Za-z])")
_APOSTROPHE_S = re.compile(r"\b([A-Za-z]+)'s\b(?!\s+(?:been|gone|got)\b)")
_OCLOCK = re.compile(r"\s*\bo'?clock\b", re.IGNORECASE)
_CLOCK_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DOTTED_CLOCK_TIME = re.compile(
    r"(?<![$£€.\d])\b(\d{1,2})\.(\d{2})(?=\s*[ap]\.?\s?m\b)", re.IGNORECASE
)
_SPLIT_MERIDIEM = re.compile(r"(?<=\d) ([ap]) m\b")

# 's after these is a contraction of "is"; after anything else it's a possessive
_CONTRACTION_HOSTS = frozenset(
    {
        "it",
        "that",
        "he",
        "she",
        "there",
        "here",
        "what",
        "who",
        "where",
        "how",
        "when",
        "why",
        "let",
    }
)

_SMALL_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    "sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
# the whisper normalizer glues digits onto a following spelled-out number
_DIGITS_THEN_NUMBER_WORD = re.compile(
    rf"(?<=\d)[,\s]+(?=(?:{_SMALL_NUMBER_WORDS})\b)", re.IGNORECASE
)

# Equivalent spellings, applied after normalization. Add to this as new ones turn up.
_VARIANTS = {
    "ok": "okay",
    "cannot": "can not",
    "all right": "alright",
    "adviser": "advisor",
    "mosquitos": "mosquitoes",
    "e mail": "email",
    "wi fi": "wifi",
    "t shirt": "tshirt",
    "re key": "rekey",
    "pre registration": "preregistration",
    "co operatively": "cooperatively",
    "inter relationships": "interrelationships",
    "micro expressions": "microexpressions",
    "non stick": "nonstick",
    "non stop": "nonstop",
    "non verbal": "nonverbal",
    "non fiction": "nonfiction",
    "team work": "teamwork",
    "no where": "nowhere",
    "dragon flies": "dragonflies",
    "tuna fish": "tunafish",
    "life cycle": "lifecycle",
    "south west": "southwest",
    "health care": "healthcare",
    "skin care": "skincare",
    "smart watch": "smartwatch",
    "light bulb": "lightbulb",
    "push ups": "pushups",
    "ride share": "rideshare",
    "story book": "storybook",
    "fire fighting": "firefighting",
    "tie break": "tiebreak",
}
_VARIANT = re.compile(r"\b(?:" + "|".join(map(re.escape, _VARIANTS)) + r")\b")


def _join_initialism(match: re.Match[str]) -> str:
    letters = match.group(0).replace(".", "")
    return letters if len(letters) > 1 else match.group(0)


def _drop_possessive_apostrophe(match: re.Match[str]) -> str:
    if match.group(1).lower() in _CONTRACTION_HOSTS:
        return match.group(0)
    return f"{match.group(1)}s"


def _join_clock_time(match: re.Match[str]) -> str:
    hour, minute = match.groups()
    return hour if minute == "00" else hour + minute


def normalize_text(text: str) -> str:
    """Canonical normalization: ``EnglishTextNormalizer`` plus our cleanup rules."""
    text = text.replace("’", "'")
    text = _ELLIPSIS.sub(" ", text)
    text = _INITIALISM.sub(_join_initialism, text)
    text = _APOSTROPHE_S.sub(_drop_possessive_apostrophe, text)
    text = _OCLOCK.sub("", text)
    text = _CLOCK_TIME.sub(_join_clock_time, text)
    text = _DOTTED_CLOCK_TIME.sub(_join_clock_time, text)
    words = [
        word
        for part in _DIGITS_THEN_NUMBER_WORD.split(text)
        for word in _normalizer(part).split()
        if any(char.isalnum() for char in word)
    ]
    normalized = _SPLIT_MERIDIEM.sub(r" \1m", " ".join(words))
    return _VARIANT.sub(lambda match: _VARIANTS[match.group(0)], normalized)


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------


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
        :class:`WERResult` with ``wer`` in [0, ∞), ``wer_percentage``, the
        jiwer alignment counts, and both normalized strings.

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
            normalized_reference=norm_ref,
            normalized_hypothesis=norm_hyp,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_words=0,
        )

    # Empty reference with non-empty hypothesis: jiwer would return inf.
    # We return the insertion count / max(1, hyp_len) so callers always get
    # a finite float. (This matches the convention documented in test_wer.py.)
    if not ref_words:
        ins_count = len(hyp_words)
        empty_ref_wer = float(ins_count)
        return WERResult(
            wer=empty_ref_wer,
            wer_percentage=empty_ref_wer * 100.0,
            normalized_reference=norm_ref,
            normalized_hypothesis=norm_hyp,
            substitutions=0,
            deletions=0,
            insertions=ins_count,
            reference_words=0,
        )

    # jiwer is the sole alignment: the ratio and the S/I/D counts come from it.
    aligned = jiwer.process_words(norm_ref, norm_hyp)
    raw_wer: float = aligned.wer

    return WERResult(
        wer=raw_wer,
        wer_percentage=raw_wer * 100.0,
        normalized_reference=norm_ref,
        normalized_hypothesis=norm_hyp,
        substitutions=aligned.substitutions,
        deletions=aligned.deletions,
        insertions=aligned.insertions,
        reference_words=len(ref_words),
    )
