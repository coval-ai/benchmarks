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


class WERResult(BaseModel):
    wer: float
    wer_percentage: float
    normalized_reference: str
    normalized_hypothesis: str
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    norm_version: Literal["2"] = NORM_VERSION

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


def normalize_text(text: str) -> str:
    """Canonical normalization: ``whisper_normalizer``'s ``EnglishTextNormalizer``."""
    normalized: str = _normalizer(text)
    return normalized


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
