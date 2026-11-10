# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Character Error Rate over the same normalized text WER is scored on."""

from __future__ import annotations

import jiwer
from pydantic import BaseModel

from coval_bench.metrics.wer import normalize_text


class CERResult(BaseModel):
    cer: float
    cer_percentage: float
    normalized_reference: str
    normalized_hypothesis: str
    errors: int
    reference_chars: int


def compute_cer(reference: str, hypothesis: str) -> CERResult:
    """Character edits over reference characters, spaces included.

    An empty reference follows ``compute_wer``: the hypothesis length is
    returned as a finite value in place of a division by zero.
    """
    norm_ref = normalize_text(reference).strip()
    norm_hyp = normalize_text(hypothesis).strip()

    if not norm_ref:
        errors = len(norm_hyp)
        cer = float(errors)
    else:
        aligned = jiwer.process_characters(norm_ref, norm_hyp)
        errors = aligned.substitutions + aligned.deletions + aligned.insertions
        cer = aligned.cer

    return CERResult(
        cer=cer,
        cer_percentage=cer * 100.0,
        normalized_reference=norm_ref,
        normalized_hypothesis=norm_hyp,
        errors=errors,
        reference_chars=len(norm_ref),
    )
