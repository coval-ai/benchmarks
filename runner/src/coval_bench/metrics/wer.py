# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Word Error Rate computation.

Text normalization is delegated to ``whisper_normalizer``'s
``EnglishTextNormalizer``, the de facto standard for published WER; the DP
edit-distance is delegated to ``jiwer`` for correctness and speed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jiwer
from pydantic import BaseModel
from whisper_normalizer.english import EnglishTextNormalizer

# Bumped on any behavioural change to normalization; surfaced on ``WERResult``
# so a result row can be attributed to the revision that produced it.
# Documented in ``docs/methodology.md`` and ADR-021.
NORM_VERSION: Literal["2"] = "2"

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
    norm_version: Literal["2"] = NORM_VERSION


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """Canonical normalization: ``whisper_normalizer``'s ``EnglishTextNormalizer``."""
    normalized: str = _normalizer(text)
    return normalized


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
