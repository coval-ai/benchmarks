# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""coval_bench.runner — orchestration layer.

Exports the primary entrypoint `run_benchmarks` and the `RunSummary` result model.
"""

from __future__ import annotations

from coval_bench.runner.config import DEFAULT_STT_MATRIX, DEFAULT_TTS_MATRIX, ProviderEntry
from coval_bench.runner.orchestrator import RunSummary, run_benchmarks

__all__ = [
    "DEFAULT_STT_MATRIX",
    "DEFAULT_TTS_MATRIX",
    "ProviderEntry",
    "RunSummary",
    "run_benchmarks",
]
