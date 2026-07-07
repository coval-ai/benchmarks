# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry of benchmark metrics and their display metadata.

``Metric`` values are the canonical strings stored in
``benchmarks_v2.results.metric_type``. ``METRIC_SPECS`` carries the display
metadata (units, ranking direction, decimals, applicable benchmarks) consumed
app-side against already-aggregated rows; the database stores plain strings.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from coval_bench.registries.benchmarks import Benchmark


class Metric(StrEnum):
    """Canonical metric identifiers as stored in ``results.metric_type``."""

    WER = "WER"
    TTFT = "TTFT"
    TTFS = "TTFS"
    TTFA = "TTFA"
    RTF = "RTF"
    AUDIO_TO_FINAL = "AudioToFinal"


class MetricDirection(StrEnum):
    """Whether smaller or larger values rank better."""

    LOWER_IS_BETTER = "lower"
    HIGHER_IS_BETTER = "higher"


class MetricSpec(BaseModel, frozen=True):
    """Display metadata for one metric."""

    display_name: str
    units: str
    direction: MetricDirection
    decimals: int
    benchmarks: frozenset[Benchmark]


# ``units`` values must stay byte-identical to what the orchestrator has
# always written to ``results.metric_units``; they are stored, not display-only.
METRIC_SPECS: dict[Metric, MetricSpec] = {
    Metric.WER: MetricSpec(
        display_name="Word Error Rate",
        units="percent",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=1,
        benchmarks=frozenset({Benchmark.STT, Benchmark.TTS}),
    ),
    Metric.TTFT: MetricSpec(
        display_name="Time to First Token",
        units="seconds",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=2,
        benchmarks=frozenset({Benchmark.STT}),
    ),
    Metric.TTFS: MetricSpec(
        display_name="Time to Final from Speech",
        units="seconds",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=2,
        benchmarks=frozenset({Benchmark.STT}),
    ),
    Metric.TTFA: MetricSpec(
        display_name="Time to First Audio",
        units="milliseconds",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=0,
        benchmarks=frozenset({Benchmark.TTS}),
    ),
    Metric.RTF: MetricSpec(
        display_name="Real-Time Factor",
        units="ratio",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=2,
        benchmarks=frozenset({Benchmark.STT}),
    ),
    Metric.AUDIO_TO_FINAL: MetricSpec(
        display_name="Audio to Final",
        units="seconds",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=2,
        benchmarks=frozenset({Benchmark.STT}),
    ),
}

if METRIC_SPECS.keys() != set(Metric):
    _missing = ", ".join(sorted(set(Metric) - METRIC_SPECS.keys()))
    raise RuntimeError(f"METRIC_SPECS is missing specs for: {_missing}")


# (provider, model) pairs whose metric is not comparable with the cohort:
# TTFT gated by buffering rather than engine speed, TTFS acked without
# finalizing. The orchestrator skips writing these rows; the API hides
# historical ones.
METRIC_EXCLUSIONS: dict[Metric, frozenset[tuple[str, str]]] = {
    Metric.TTFT: frozenset(
        {
            ("xai", "grok-stt"),
            ("openai", "gpt-4o-transcribe"),
            ("openai", "gpt-4o-mini-transcribe"),
        }
    ),
    Metric.TTFS: frozenset(
        {
            ("deepgram", "flux-general-en"),
            ("deepgram", "flux-general-multi"),
            ("assemblyai", "universal-streaming"),
            ("assemblyai", "universal-streaming-multilingual"),
        }
    ),
}


def is_metric_excluded(provider: str, model: str, metric: str) -> bool:
    """True if this (provider, model) pair is excluded from ``metric``."""
    try:
        pairs = METRIC_EXCLUSIONS.get(Metric(metric))
    except ValueError:
        return False
    return pairs is not None and (provider, model) in pairs
