# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic semantic-unit scoring over WER token alignments.

This module is intentionally standalone. It scores annotated reference spans
and does not infer semantics, call a model, or assign hypothesis-only insertions.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from collections.abc import Iterable
from typing import Literal

import jiwer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from coval_bench.metrics.wer import NORM_VERSION, WERResult, compute_wer, normalize_text

SEMANTIC_WER_VERSION: Literal["0.1-draft"] = "0.1-draft"

SemanticLevel = Literal[1, 2, 3]
SemanticUnitKind = Literal[
    "action",
    "entity",
    "quantity",
    "datetime",
    "money",
    "identifier",
    "attribute",
    "relation",
    "polarity",
    "destination",
    "topic",
]
SemanticUnitOutcome = Literal["correct", "substituted", "deleted"]
LEVEL_WEIGHTS: dict[SemanticLevel, int] = {1: 0, 2: 1, 3: 3}
_POLARITY_TOKENS = frozenset({"not", "no", "never", "without", "except"})
_IDENTIFIER_DASH_MARKER = "covaliddashmarker"
_IDENTIFIER_STAR_MARKER = "covalidstarmarker"
_IDENTIFIER_LETTER_NAMES = {
    "are": "r",
    "bee": "b",
    "kay": "k",
    "why": "y",
}


class SemanticComponent(BaseModel):
    """A minimal required span within one logical semantic unit."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    accepted_forms: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_component(self) -> SemanticComponent:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if any(not form.strip() for form in self.accepted_forms):
            raise ValueError("accepted_forms must not contain blank values")
        if len(self.accepted_forms) != len(set(self.accepted_forms)):
            raise ValueError("accepted_forms must be unique")
        return self


class SemanticUnit(BaseModel):
    """A meaning-bearing unit in the reference."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: SemanticUnitKind
    role: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    canonical_value: JsonValue
    level: SemanticLevel
    accepted_forms: tuple[str, ...] = ()
    components: tuple[SemanticComponent, ...] = ()

    @model_validator(mode="after")
    def _end_follows_start(self) -> SemanticUnit:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if self.canonical_value is None:
            raise ValueError("canonical_value must not be null")
        if any(not form.strip() for form in self.accepted_forms):
            raise ValueError("accepted_forms must not contain blank values")
        if len(self.accepted_forms) != len(set(self.accepted_forms)):
            raise ValueError("accepted_forms must be unique")
        if self.kind == "identifier" and self.components:
            raise ValueError("identifier units cannot define components")
        if self.components and self.accepted_forms:
            raise ValueError("componentized units cannot also define unit-level accepted_forms")

        component_ids: set[str] = set()
        previous_end = self.start_char
        for component in sorted(
            self.components,
            key=lambda candidate: (candidate.start_char, candidate.end_char),
        ):
            if component.id in component_ids:
                raise ValueError(f"duplicate semantic component id: {component.id}")
            component_ids.add(component.id)
            if component.start_char < self.start_char or component.end_char > self.end_char:
                raise ValueError(f"semantic component {component.id!r} exceeds its unit span")
            if component.start_char < previous_end:
                raise ValueError(f"semantic component {component.id!r} overlaps another component")
            relative_start = component.start_char - self.start_char
            relative_end = component.end_char - self.start_char
            if self.text[relative_start:relative_end] != component.text:
                raise ValueError(
                    f"semantic component {component.id!r} text does not match its unit span"
                )
            previous_end = component.end_char
        return self


class SemanticUnitScore(BaseModel):
    unit_id: str
    kind: SemanticUnitKind
    role: str
    level: SemanticLevel
    weight: int
    reference_text: str
    hypothesis_text: str | None
    outcome: SemanticUnitOutcome


class SemanticWERResult(BaseModel):
    semantic_unit_error_rate: float
    core_unit_error_rate: float
    qualifier_unit_error_rate: float
    semantic_utterance_success: bool
    error_weight: int
    reference_weight: int
    core_errors: int
    core_units: int
    qualifier_errors: int
    qualifier_units: int
    surface_errors: int
    surface_units: int
    unit_scores: list[SemanticUnitScore]
    wer: WERResult
    semantic_wer_version: Literal["0.1-draft"] = SEMANTIC_WER_VERSION
    norm_version: Literal["2"] = NORM_VERSION

    def normalized_value_rows(self) -> list[SemanticValueRow]:
        """Return sufficient statistics for one benchmark observation."""
        return _semantic_value_rows(
            semantic_unit_error_rate=self.semantic_unit_error_rate,
            error_weight=self.error_weight,
            reference_weight=self.reference_weight,
            core_errors=self.core_errors,
            core_units=self.core_units,
            qualifier_errors=self.qualifier_errors,
            qualifier_units=self.qualifier_units,
            surface_errors=self.surface_errors,
            surface_units=self.surface_units,
            semantic_successes=int(self.semantic_utterance_success),
            utterances=1,
        )


class SemanticValueRow(BaseModel):
    """Platform-neutral normalized value emitted by the library scorer."""

    metric: Literal["SemanticWER"] = "SemanticWER"
    metric_version: Literal["0.1-draft"] = SEMANTIC_WER_VERSION
    value_key: str = Field(min_length=1)
    unit: Literal["percent", "weight", "count"]
    value: float
    value_role: Literal["primary", "component"]


ValueUnit = Literal["percent", "weight", "count"]
ValueRole = Literal["primary", "component"]


def _semantic_value_rows(
    *,
    semantic_unit_error_rate: float,
    error_weight: int,
    reference_weight: int,
    core_errors: int,
    core_units: int,
    qualifier_errors: int,
    qualifier_units: int,
    surface_errors: int,
    surface_units: int,
    semantic_successes: int,
    utterances: int,
) -> list[SemanticValueRow]:
    values: tuple[tuple[str, ValueUnit, float, ValueRole], ...] = (
        ("primary", "percent", semantic_unit_error_rate * 100, "primary"),
        ("error_weight", "weight", error_weight, "component"),
        ("reference_weight", "weight", reference_weight, "component"),
        ("core_errors", "count", core_errors, "component"),
        ("core_units", "count", core_units, "component"),
        ("qualifier_errors", "count", qualifier_errors, "component"),
        ("qualifier_units", "count", qualifier_units, "component"),
        ("surface_errors", "count", surface_errors, "component"),
        ("surface_units", "count", surface_units, "component"),
        ("semantic_successes", "count", semantic_successes, "component"),
        ("utterances", "count", utterances, "component"),
    )
    return [
        SemanticValueRow(
            value_key=value_key,
            unit=unit,
            value=value,
            value_role=value_role,
        )
        for value_key, unit, value, value_role in values
    ]


class SemanticWERAggregate(BaseModel):
    semantic_unit_error_rate: float
    core_unit_error_rate: float
    qualifier_unit_error_rate: float
    semantic_utterance_success_rate: float
    wer: float
    samples: int
    error_weight: int
    reference_weight: int
    core_errors: int
    core_units: int
    qualifier_errors: int
    qualifier_units: int
    surface_errors: int
    surface_units: int
    semantic_successes: int
    utterances: int
    outcome_counts_by_kind: dict[str, dict[str, int]]
    semantic_wer_version: Literal["0.1-draft"] = SEMANTIC_WER_VERSION
    norm_version: Literal["2"] = NORM_VERSION

    def normalized_value_rows(self) -> list[SemanticValueRow]:
        """Return pooled values without exposing per-sample ratios as aggregates."""
        return _semantic_value_rows(
            semantic_unit_error_rate=self.semantic_unit_error_rate,
            error_weight=self.error_weight,
            reference_weight=self.reference_weight,
            core_errors=self.core_errors,
            core_units=self.core_units,
            qualifier_errors=self.qualifier_errors,
            qualifier_units=self.qualifier_units,
            surface_errors=self.surface_errors,
            surface_units=self.surface_units,
            semantic_successes=self.semantic_successes,
            utterances=self.utterances,
        )


def _validate_units(reference: str, units: list[SemanticUnit]) -> None:
    seen_ids: set[str] = set()
    previous_end = -1

    for unit in sorted(units, key=lambda candidate: (candidate.start_char, candidate.end_char)):
        if unit.id in seen_ids:
            raise ValueError(f"duplicate semantic unit id: {unit.id}")
        seen_ids.add(unit.id)

        if unit.end_char > len(reference):
            raise ValueError(f"semantic unit {unit.id!r} extends past the reference")
        actual_text = reference[unit.start_char : unit.end_char]
        if actual_text != unit.text:
            raise ValueError(
                f"semantic unit {unit.id!r} text does not match its reference span: "
                f"expected {unit.text!r}, found {actual_text!r}"
            )
        if unit.start_char < previous_end:
            raise ValueError(f"semantic unit {unit.id!r} overlaps the previous unit")
        previous_end = unit.end_char


class SemanticSampleAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    semantic_units: list[SemanticUnit] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_reference_units(self) -> SemanticSampleAnnotation:
        _validate_units(self.reference, self.semantic_units)
        return self


class SemanticAnnotationSet(BaseModel):
    """Neutral public annotation schema; review/provenance belongs to private harnesses."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1-draft"]
    dataset_id: str = Field(min_length=1)
    dataset_identity: str = Field(min_length=1)
    normalizer_version: Literal["2"]
    samples: list[SemanticSampleAnnotation] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_samples(self) -> SemanticAnnotationSet:
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id values must be unique within an annotation set")

        return self

    def sample(self, sample_id: str) -> SemanticSampleAnnotation:
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        raise KeyError(f"unknown semantic annotation sample: {sample_id}")


def _normalized_reference_span(
    reference: str,
    span: SemanticUnit | SemanticComponent,
) -> tuple[int, int]:
    """Map a raw character span to its token span after WER normalization."""
    token_start = len(normalize_text(reference[: span.start_char]).split())
    token_end = len(normalize_text(reference[: span.end_char]).split())
    normalized_unit = normalize_text(span.text).split()
    normalized_reference = normalize_text(reference).split()

    if not normalized_unit:
        raise ValueError(f"semantic span {span.id!r} is empty after normalization")
    if normalized_reference[token_start:token_end] != normalized_unit:
        raise ValueError(
            f"semantic span {span.id!r} is not stable under normalization; "
            "use a span whose normalized tokens match the reference slice"
        )
    return token_start, token_end


def _aligned_hypothesis_candidates(
    alignments: list[jiwer.AlignmentChunk],
    hypothesis_tokens: list[str],
    reference_start: int,
    reference_end: int,
) -> list[list[str]]:
    aligned: list[str] = []
    leading: list[str] = []
    trailing: list[str] = []
    for chunk in alignments:
        if chunk.type == "insert":
            inserted = hypothesis_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx]
            if chunk.ref_start_idx == reference_start:
                leading.extend(inserted)
            elif chunk.ref_start_idx == reference_end:
                trailing.extend(inserted)
            elif reference_start < chunk.ref_start_idx < reference_end:
                aligned.extend(inserted)
            continue

        overlap_start = max(reference_start, chunk.ref_start_idx)
        overlap_end = min(reference_end, chunk.ref_end_idx)
        if overlap_start >= overlap_end or chunk.type == "delete":
            continue

        if chunk.type == "equal":
            hyp_start = chunk.hyp_start_idx + (overlap_start - chunk.ref_start_idx)
            hyp_end = hyp_start + (overlap_end - overlap_start)
            aligned.extend(hypothesis_tokens[hyp_start:hyp_end])
        else:
            if (chunk.ref_end_idx - chunk.ref_start_idx) == (
                chunk.hyp_end_idx - chunk.hyp_start_idx
            ):
                hyp_start = chunk.hyp_start_idx + (overlap_start - chunk.ref_start_idx)
                hyp_end = hyp_start + (overlap_end - overlap_start)
                aligned.extend(hypothesis_tokens[hyp_start:hyp_end])
            else:
                # Different token counts have no unambiguous positional map.
                aligned.extend(hypothesis_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx])

    candidates = [aligned]
    if leading:
        candidates.append([*leading, *aligned])
    if trailing:
        candidates.append([*aligned, *trailing])
    if leading and trailing:
        candidates.append([*leading, *aligned, *trailing])
    return candidates


def canonicalize_identifier_text(text: str) -> str:
    """Canonicalize common ASR renderings while preserving ID symbols."""
    prepared = text.replace("-", f" {_IDENTIFIER_DASH_MARKER} ")
    prepared = prepared.replace("*", f" {_IDENTIFIER_STAR_MARKER} ")
    prepared = re.sub(
        r"\b(?:dash|hyphen)\b",
        f" {_IDENTIFIER_DASH_MARKER} ",
        prepared,
        flags=re.IGNORECASE,
    )
    prepared = re.sub(
        r"\b(?:star|asterisk)\b",
        f" {_IDENTIFIER_STAR_MARKER} ",
        prepared,
        flags=re.IGNORECASE,
    )
    for spoken, letter in _IDENTIFIER_LETTER_NAMES.items():
        prepared = re.sub(rf"\b{spoken}\b", letter, prepared, flags=re.IGNORECASE)

    normalized = normalize_text(prepared)
    normalized = normalized.replace(_IDENTIFIER_DASH_MARKER, "-")
    normalized = normalized.replace(_IDENTIFIER_STAR_MARKER, "*")
    return re.sub(r"[^a-z0-9*-]", "", normalized).upper()


def _aligned_raw_identifier_candidates(
    hypothesis: str,
    normalized_hypothesis: str,
    alignments: list[jiwer.AlignmentChunk],
    reference_start: int,
    reference_end: int,
) -> list[str]:
    """Return raw identifier candidates from only the aligned hypothesis span."""
    raw_tokens = hypothesis.split()
    normalized_targets = normalized_hypothesis.split()

    @functools.cache
    def partition(raw_index: int, normalized_index: int) -> tuple[tuple[int, ...], ...] | None:
        if normalized_index == len(normalized_targets):
            return () if raw_index == len(raw_tokens) else None
        for raw_end in range(raw_index + 1, len(raw_tokens) + 1):
            raw_span = " ".join(raw_tokens[raw_index:raw_end])
            normalized_span = normalize_text(raw_span).split()
            span_length = len(normalized_span)
            if (
                normalized_targets[normalized_index : normalized_index + span_length]
                != normalized_span
            ):
                continue
            remainder = partition(raw_end, normalized_index + span_length)
            if remainder is not None:
                return ((tuple(range(raw_index, raw_end))), *remainder)
        return None

    raw_groups = partition(0, 0)
    if raw_groups is None:
        return []
    raw_map: list[tuple[int, ...]] = []
    for group in raw_groups:
        span_length = len(normalize_text(" ".join(raw_tokens[index] for index in group)).split())
        raw_map.extend([group] * span_length)
    if len(raw_map) != len(normalized_targets):
        return []
    aligned: list[int] = []
    leading: list[int] = []
    trailing: list[int] = []
    for chunk in alignments:
        if chunk.type == "insert":
            inserted = [
                raw_index
                for mapped in raw_map[chunk.hyp_start_idx : chunk.hyp_end_idx]
                for raw_index in mapped
            ]
            if chunk.ref_start_idx == reference_start:
                leading.extend(inserted)
            elif chunk.ref_start_idx == reference_end:
                trailing.extend(inserted)
            elif reference_start < chunk.ref_start_idx < reference_end:
                aligned.extend(inserted)
            continue
        overlap_start = max(reference_start, chunk.ref_start_idx)
        overlap_end = min(reference_end, chunk.ref_end_idx)
        if overlap_start >= overlap_end or chunk.type == "delete":
            continue
        if chunk.type == "equal" or (
            chunk.ref_end_idx - chunk.ref_start_idx == chunk.hyp_end_idx - chunk.hyp_start_idx
        ):
            hyp_start = chunk.hyp_start_idx + (overlap_start - chunk.ref_start_idx)
            hyp_end = hyp_start + (overlap_end - overlap_start)
            aligned.extend(
                raw_index for mapped in raw_map[hyp_start:hyp_end] for raw_index in mapped
            )
        else:
            aligned.extend(
                raw_index
                for mapped in raw_map[chunk.hyp_start_idx : chunk.hyp_end_idx]
                for raw_index in mapped
            )

    def raw_candidate(indices: list[int]) -> str:
        if not indices:
            return ""
        start = min(indices)
        end = max(indices) + 1

        def is_identifier_symbol(token: str) -> bool:
            canonical = canonicalize_identifier_text(token)
            return bool(canonical) and all(character in "-*" for character in canonical)

        while start > 0 and is_identifier_symbol(raw_tokens[start - 1]):
            start -= 1
        while end < len(raw_tokens) and is_identifier_symbol(raw_tokens[end]):
            end += 1
        return " ".join(raw_tokens[start:end])

    candidates = [aligned]
    if leading:
        candidates.append([*leading, *aligned])
    if trailing:
        candidates.append([*aligned, *trailing])
    if leading and trailing:
        candidates.append([*leading, *aligned, *trailing])
    return [raw_candidate(candidate) for candidate in candidates]


def _score_span(
    reference: str,
    span: SemanticUnit | SemanticComponent,
    alignments: list[jiwer.AlignmentChunk],
    hypothesis_tokens: list[str],
) -> tuple[SemanticUnitOutcome, str | None]:
    reference_start, reference_end = _normalized_reference_span(reference, span)
    candidate_options = _aligned_hypothesis_candidates(
        alignments,
        hypothesis_tokens,
        reference_start,
        reference_end,
    )
    candidate_tokens = candidate_options[0]
    candidate = " ".join(candidate_tokens)
    accepted = {
        normalize_text(form) for form in (span.text, *span.accepted_forms) if normalize_text(form)
    }

    matched_candidate = next(
        (" ".join(option) for option in candidate_options if " ".join(option) in accepted),
        None,
    )

    outcome: SemanticUnitOutcome
    if matched_candidate is not None:
        outcome = "correct"
        hypothesis_text = matched_candidate
    elif not candidate_tokens:
        outcome = "deleted"
        hypothesis_text = None
    else:
        outcome = "substituted"
        hypothesis_text = candidate
    return outcome, hypothesis_text


def _score_unit(
    reference: str,
    hypothesis: str,
    unit: SemanticUnit,
    alignments: list[jiwer.AlignmentChunk],
    hypothesis_tokens: list[str],
) -> SemanticUnitScore:
    outcome: SemanticUnitOutcome
    hypothesis_text: str | None
    if unit.kind == "identifier":
        if not isinstance(unit.canonical_value, str):
            raise ValueError(f"identifier unit {unit.id!r} requires a string canonical_value")
        canonical_target = canonicalize_identifier_text(unit.canonical_value)
        if not canonical_target:
            raise ValueError(f"identifier unit {unit.id!r} requires a non-empty canonical_value")
        allowed_targets = {
            canonicalize_identifier_text(form)
            for form in (unit.canonical_value, unit.text, *unit.accepted_forms)
        }
        allowed_targets.discard("")
        reference_start, reference_end = _normalized_reference_span(reference, unit)
        raw_candidates = _aligned_raw_identifier_candidates(
            hypothesis,
            normalize_text(hypothesis),
            alignments,
            reference_start,
            reference_end,
        )
        expanded_candidates = raw_candidates[1:] if len(raw_candidates) > 1 else raw_candidates
        matching = next(
            (
                candidate
                for candidate in expanded_candidates
                if canonicalize_identifier_text(candidate) in allowed_targets
            ),
            None,
        )
        if matching is None and raw_candidates:
            base = raw_candidates[0]
            base_tokens = base.split()
            extras: list[str] = []
            for candidate in raw_candidates[1:]:
                candidate_tokens = candidate.split()
                if candidate_tokens[: len(base_tokens)] == base_tokens:
                    extras.extend(candidate_tokens[len(base_tokens) :])
                elif candidate_tokens[-len(base_tokens) :] == base_tokens:
                    extras.extend(candidate_tokens[: -len(base_tokens)])
            if canonicalize_identifier_text(base) in allowed_targets and not any(
                any(char.isdigit() for char in canonicalize_identifier_text(token))
                or "-" in canonicalize_identifier_text(token)
                or "*" in canonicalize_identifier_text(token)
                or len(canonicalize_identifier_text(token)) == 1
                for token in extras
            ):
                matching = base
        if matching is not None:
            outcome = "correct"
            hypothesis_text = canonical_target
        elif not any(raw_candidates):
            outcome = "deleted"
            hypothesis_text = None
        else:
            outcome = "substituted"
            hypothesis_text = next((candidate for candidate in raw_candidates if candidate), None)
    elif unit.components:
        component_scores = [
            _score_span(reference, component, alignments, hypothesis_tokens)
            for component in unit.components
        ]
        component_outcomes = [outcome for outcome, _ in component_scores]
        if all(outcome == "correct" for outcome in component_outcomes):
            outcome = "correct"
        elif all(outcome == "deleted" for outcome in component_outcomes):
            outcome = "deleted"
        else:
            outcome = "substituted"

        reference_start, reference_end = _normalized_reference_span(reference, unit)
        envelope = _aligned_hypothesis_candidates(
            alignments,
            hypothesis_tokens,
            reference_start,
            reference_end,
        )[0]
        reference_polarity = _POLARITY_TOKENS.intersection(normalize_text(unit.text).split())
        hypothesis_polarity = _POLARITY_TOKENS.intersection(envelope)
        if hypothesis_polarity != reference_polarity:
            outcome = "substituted"

        recognized = [text for _, text in component_scores if text]
        hypothesis_text = " … ".join(recognized) if recognized else None
    else:
        outcome, hypothesis_text = _score_span(
            reference,
            unit,
            alignments,
            hypothesis_tokens,
        )

    return SemanticUnitScore(
        unit_id=unit.id,
        kind=unit.kind,
        role=unit.role,
        level=unit.level,
        weight=LEVEL_WEIGHTS[unit.level],
        reference_text=unit.text,
        hypothesis_text=hypothesis_text,
        outcome=outcome,
    )


def compute_semantic_wer(
    reference: str,
    hypothesis: str,
    units: Iterable[SemanticUnit],
) -> SemanticWERResult:
    """Preview or test reference units against a hypothesis deterministically.

    Version ``0.1-draft`` does not score insertions as independent semantic
    units because no reference span can type or weight them without another
    semantic inference step. Alignment can still absorb an adjacent insertion
    into a reviewed unit and penalize that unit.
    """
    unit_list = list(units)
    _validate_units(reference, unit_list)

    wer = compute_wer(reference, hypothesis)
    processed = jiwer.process_words(wer.normalized_reference, wer.normalized_hypothesis)
    alignments = processed.alignments[0]
    hypothesis_tokens = wer.normalized_hypothesis.split()
    scores = [
        _score_unit(reference, hypothesis, unit, alignments, hypothesis_tokens)
        for unit in unit_list
    ]

    errors = [score for score in scores if score.outcome != "correct"]
    reference_weight = sum(score.weight for score in scores)
    error_weight = sum(score.weight for score in errors)
    core_units = sum(score.level == 3 for score in scores)
    core_errors = sum(score.level == 3 for score in errors)
    qualifier_units = sum(score.level == 2 for score in scores)
    qualifier_errors = sum(score.level == 2 for score in errors)
    surface_units = sum(score.level == 1 for score in scores)
    surface_errors = sum(score.level == 1 for score in errors)

    return SemanticWERResult(
        semantic_unit_error_rate=error_weight / reference_weight if reference_weight else 0.0,
        core_unit_error_rate=core_errors / core_units if core_units else 0.0,
        qualifier_unit_error_rate=(qualifier_errors / qualifier_units if qualifier_units else 0.0),
        semantic_utterance_success=core_errors == 0,
        error_weight=error_weight,
        reference_weight=reference_weight,
        core_errors=core_errors,
        core_units=core_units,
        qualifier_errors=qualifier_errors,
        qualifier_units=qualifier_units,
        surface_errors=surface_errors,
        surface_units=surface_units,
        unit_scores=scores,
        wer=wer,
    )


def aggregate_semantic_wer(results: Iterable[SemanticWERResult]) -> SemanticWERAggregate:
    """Pool counts across utterances; never average per-utterance ratios."""
    result_list = list(results)
    samples = len(result_list)
    reference_weight = sum(result.reference_weight for result in result_list)
    error_weight = sum(result.error_weight for result in result_list)
    core_units = sum(result.core_units for result in result_list)
    core_errors = sum(result.core_errors for result in result_list)
    qualifier_units = sum(result.qualifier_units for result in result_list)
    qualifier_errors = sum(result.qualifier_errors for result in result_list)
    surface_units = sum(result.surface_units for result in result_list)
    surface_errors = sum(result.surface_errors for result in result_list)
    wer_reference_words = sum(result.wer.reference_words for result in result_list)
    wer_errors = sum(
        result.wer.substitutions + result.wer.deletions + result.wer.insertions
        for result in result_list
    )

    by_kind: dict[str, Counter[str]] = {}
    for result in result_list:
        for score in result.unit_scores:
            by_kind.setdefault(score.kind, Counter())[score.outcome] += 1

    return SemanticWERAggregate(
        semantic_unit_error_rate=error_weight / reference_weight if reference_weight else 0.0,
        core_unit_error_rate=core_errors / core_units if core_units else 0.0,
        qualifier_unit_error_rate=(qualifier_errors / qualifier_units if qualifier_units else 0.0),
        semantic_utterance_success_rate=(
            sum(result.semantic_utterance_success for result in result_list) / samples
            if samples
            else 0.0
        ),
        wer=wer_errors / wer_reference_words if wer_reference_words else float(wer_errors),
        samples=samples,
        error_weight=error_weight,
        reference_weight=reference_weight,
        core_errors=core_errors,
        core_units=core_units,
        qualifier_errors=qualifier_errors,
        qualifier_units=qualifier_units,
        surface_errors=surface_errors,
        surface_units=surface_units,
        semantic_successes=sum(result.semantic_utterance_success for result in result_list),
        utterances=samples,
        outcome_counts_by_kind={kind: dict(counts) for kind, counts in by_kind.items()},
    )
