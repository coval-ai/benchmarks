# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Public API for coval_bench metrics."""

from coval_bench.metrics.ttfa import compute_ttfa, first_audible_offset_ms
from coval_bench.metrics.ttft import compute_rtf, compute_ttfs
from coval_bench.metrics.wer import WERResult, WordError, compute_wer, normalize_text

__all__ = [
    "WERResult",
    "WordError",
    "compute_wer",
    "normalize_text",
    "compute_ttfa",
    "first_audible_offset_ms",
    "compute_rtf",
    "compute_ttfs",
]
