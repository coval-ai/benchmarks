# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""The benchmarks themselves — the top-level axis every registry keys on."""

from __future__ import annotations

from enum import StrEnum


class Benchmark(StrEnum):
    """Which benchmark a model or result belongs to."""

    STT = "STT"
    TTS = "TTS"
