# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Public API for coval_bench metrics (WER, TTFA, TTFT, RTF)."""

from coval_bench.metrics.ttfa import compute_ttfa
from coval_bench.metrics.ttft import compute_audio_to_final, compute_rtf, compute_ttft
from coval_bench.metrics.wer import WERResult, WordError, compute_wer, normalize_text

__all__ = [
    "WERResult",
    "WordError",
    "compute_wer",
    "normalize_text",
    "compute_ttfa",
    "compute_ttft",
    "compute_audio_to_final",
    "compute_rtf",
]
