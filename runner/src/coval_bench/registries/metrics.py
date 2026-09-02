# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry of benchmark metrics and their display metadata.

``Metric`` values are the canonical strings stored in
``benchmarks_v2.results.metric_type``. ``METRIC_SPECS`` carries the display
metadata (units, ranking direction, decimals, applicable benchmarks) consumed
app-side against already-aggregated rows; the database stores plain strings.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel

from coval_bench.registries.benchmarks import Benchmark


class Metric(StrEnum):
    """Canonical metric identifiers as stored in ``results.metric_type``."""

    WER = "WER"
    TTFT = "TTFT"
    TTFS = "TTFS"
    TTFA = "TTFA"
    TTFA_ROUNDTRIP = "TTFARoundtrip"
    TTFA_LEADING_SILENCE = "TTFALeadingSilence"
    RTF = "RTF"
    AUDIO_TO_FINAL = "AudioToFinal"
    V2V = "V2V"
    INSTRUCTION_FOLLOWING = "InstructionFollowing"
    INTERRUPTION_RATE = "InterruptionRate"


class MetricDirection(StrEnum):
    """Whether smaller or larger values rank better."""

    LOWER_IS_BETTER = "lower"
    HIGHER_IS_BETTER = "higher"


class MetricValueRole(StrEnum):
    """The semantic role of a value emitted by a metric evaluation."""

    PRIMARY = "primary"
    COMPONENT = "component"


class MetricSpec(BaseModel, frozen=True):
    """Display metadata for one metric."""

    display_name: str
    units: str
    direction: MetricDirection
    decimals: int
    benchmarks: frozenset[Benchmark]


class MetricValueDefinition(BaseModel, frozen=True):
    """One value in the normalized, versioned metric result contract."""

    key: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    value_role: MetricValueRole = MetricValueRole.COMPONENT
    required: bool = True


class MetricValueContract(BaseModel, frozen=True):
    """The values emitted by one metric implementation version."""

    metric: Metric
    version: str
    values: tuple[MetricValueDefinition, ...]
    component_sum_tolerance: float | None = None
    optional_all_or_none: tuple[frozenset[str], ...] = ()


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
        benchmarks=frozenset({Benchmark.STT, Benchmark.LLM}),
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
    # Perceived TTFA split: roundtrip (send → first chunk) + leading silence
    # (stream start → first audible sample). Written only when both are known,
    # so the two rows always sum back to the TTFA row.
    Metric.TTFA_ROUNDTRIP: MetricSpec(
        display_name="TTFA Network Roundtrip",
        units="milliseconds",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=0,
        benchmarks=frozenset({Benchmark.TTS}),
    ),
    Metric.TTFA_LEADING_SILENCE: MetricSpec(
        display_name="TTFA Leading Silence",
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
    Metric.V2V: MetricSpec(
        display_name="Voice-to-Voice Latency",
        units="milliseconds",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=0,
        benchmarks=frozenset({Benchmark.S2S}),
    ),
    Metric.INSTRUCTION_FOLLOWING: MetricSpec(
        display_name="Instruction Adherence",
        # Per-conversation pass stored as 100.0 (YES) / 0.0 (NO); the aggregate
        # average is the pass rate as a percentage, like WER.
        units="percent",
        direction=MetricDirection.HIGHER_IS_BETTER,
        decimals=1,
        benchmarks=frozenset({Benchmark.S2S, Benchmark.LLM}),
    ),
    Metric.INTERRUPTION_RATE: MetricSpec(
        display_name="Interruption Rate",
        units="per_minute",
        direction=MetricDirection.LOWER_IS_BETTER,
        decimals=2,
        benchmarks=frozenset({Benchmark.S2S}),
    ),
}

if METRIC_SPECS.keys() != set(Metric):
    _missing = ", ".join(sorted(set(Metric) - METRIC_SPECS.keys()))
    raise RuntimeError(f"METRIC_SPECS is missing specs for: {_missing}")


def _primary(metric: Metric) -> MetricValueDefinition:
    spec = METRIC_SPECS[metric]
    # WER can legitimately exceed 100% when insertions outnumber reference
    # words. Only the binary instruction-following rate is intrinsically bounded.
    maximum = 100.0 if metric is Metric.INSTRUCTION_FOLLOWING else None
    return MetricValueDefinition(
        key="primary",
        unit=spec.units,
        minimum=0.0,
        maximum=maximum,
        value_role=MetricValueRole.PRIMARY,
    )


# This is deliberately independent from the legacy result-column layout.  A
# metric implementation can add a new version without changing public rows.
METRIC_VALUE_CONTRACTS: dict[tuple[Metric, str], MetricValueContract] = {
    (metric, "v1"): MetricValueContract(metric=metric, version="v1", values=(_primary(metric),))
    for metric in Metric
}
METRIC_VALUE_CONTRACTS[(Metric.WER, "v1")] = MetricValueContract(
    metric=Metric.WER,
    version="v1",
    values=(
        _primary(Metric.WER),
        MetricValueDefinition(key="insertions", unit="percent", minimum=0.0),
        MetricValueDefinition(key="deletions", unit="percent", minimum=0.0),
        MetricValueDefinition(key="substitutions", unit="percent", minimum=0.0),
    ),
    component_sum_tolerance=0.0001,
)
METRIC_VALUE_CONTRACTS[(Metric.TTFA, "v1")] = MetricValueContract(
    metric=Metric.TTFA,
    version="v1",
    values=(
        _primary(Metric.TTFA),
        MetricValueDefinition(key="roundtrip", unit="milliseconds", minimum=0.0, required=False),
        MetricValueDefinition(
            key="leading_silence", unit="milliseconds", minimum=0.0, required=False
        ),
    ),
    component_sum_tolerance=0.001,
    optional_all_or_none=(frozenset({"roundtrip", "leading_silence"}),),
)


def validate_metric_contract(metric: Metric | str, version: str) -> MetricValueContract:
    """Resolve one supported metric/version contract from the application registry."""
    try:
        return METRIC_VALUE_CONTRACTS[(Metric(metric), version)]
    except ValueError as exc:
        raise ValueError(f"unknown metric {metric!r}") from exc
    except KeyError as exc:
        raise ValueError(f"unknown metric/version {metric!r}/{version!r}") from exc


def validate_metric_values(
    metric: Metric | str,
    version: str,
    values: tuple[tuple[str, str, float, MetricValueRole], ...],
) -> None:
    """Validate one normalized metric evaluation before it reaches the DB.

    Values are ``(key, unit, value, value_role)`` tuples so the persistence
    layer remains free to use its own Pydantic input models.
    """
    contract = validate_metric_contract(metric, version)
    definitions = {definition.key: definition for definition in contract.values}
    seen: set[str] = set()
    primary_count = 0
    by_key: dict[str, float] = {}
    for key, unit, value, value_role in values:
        if key in seen:
            raise ValueError(f"duplicate metric value key {key!r}")
        seen.add(key)
        definition = definitions.get(key)
        if definition is None:
            raise ValueError(f"unknown metric value key {key!r}")
        if unit != definition.unit:
            raise ValueError(f"wrong unit for {key!r}: {unit!r}")
        if value_role != definition.value_role:
            raise ValueError(f"wrong value role for {key!r}")
        if not math.isfinite(value):
            raise ValueError(f"metric value {key!r} must be finite")
        if definition.minimum is not None and value < definition.minimum:
            raise ValueError(f"metric value {key!r} is below its minimum")
        if definition.maximum is not None and value > definition.maximum:
            raise ValueError(f"metric value {key!r} is above its maximum")
        primary_count += int(value_role is MetricValueRole.PRIMARY)
        by_key[key] = value
    if primary_count != 1:
        raise ValueError("metric evaluation must contain exactly one primary value")
    required_keys = {definition.key for definition in contract.values if definition.required}
    if not required_keys <= seen:
        missing = ", ".join(sorted(required_keys - seen))
        raise ValueError(f"missing metric value keys: {missing}")
    for optional_group in contract.optional_all_or_none:
        present = seen & optional_group
        if present and present != optional_group:
            missing = ", ".join(sorted(optional_group - present))
            raise ValueError(f"optional metric value group is incomplete; missing: {missing}")
    if contract.component_sum_tolerance is not None:
        components = [value for key, value in by_key.items() if key != "primary"]
        if (
            components
            and abs(sum(components) - by_key["primary"]) > contract.component_sum_tolerance
        ):
            raise ValueError("metric components must sum to primary within tolerance")


# Metrics kept out of the per-bucket series rollup (results_by_bucket) and
# therefore out of every aggregates response's `series` array. The TTFA
# components are consumed as window aggregates only (the Latency Variation
# breakdown); TTS runs ~48x/day, so carrying them per bucket would double an
# already multi-MB 30d series payload for rows nothing reads. Remove a metric
# here if a per-run surface ever ships for it.
SERIES_EXCLUDED_METRICS: frozenset[Metric] = frozenset(
    {Metric.TTFA_ROUNDTRIP, Metric.TTFA_LEADING_SILENCE}
)


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
            # Modulate's English endpoint emits partials on a fixed ~1.5s
            # cadence, so first-token timing tracks the emission interval.
            ("modulate", "velma-2-stt-streaming-english-v2"),
            # Reson8 emits interims on a fixed ~1.1s grid, so first-token timing
            # tracks the emission interval.
            ("reson8", "realtime"),
            # Scribe emits no partials — the first token is the turn-final, so
            # first-token timing tracks utterance length.
            ("zoom", "scribe"),
        }
    ),
    Metric.TTFS: frozenset(
        {
            ("deepgram", "flux-general-en"),
            ("deepgram", "flux-general-multi"),
            ("assemblyai", "universal-streaming"),
            ("assemblyai", "universal-streaming-multilingual"),
            # Rev AI has no force-finalize; the tail final only lands after Reverb's
            # endpointer fires on trailing silence, so TTFS bundles endpoint-detection
            # latency and isn't comparable to force-endpoint providers.
            ("revai", "reverb"),
            # Together's commit drops Nemotron's encoder lookahead, so the client
            # pads trailing silence before committing; the final's timing then
            # tracks the pad length, not the engine.
            ("together", "nemotron-3-asr-streaming-0.6b"),
            ("together", "nemotron-3.5-asr-streaming-0.6b"),
            # Scribe has no force-finalize (session.close drops the pending turn);
            # the client pads trailing silence, so the final's timing bundles
            # endpoint detection.
            ("zoom", "scribe"),
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
