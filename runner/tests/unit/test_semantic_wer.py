from __future__ import annotations

import pytest

from coval_bench.metrics.semantic_wer import (
    LEVEL_WEIGHTS,
    SemanticAnnotationSet,
    SemanticComponent,
    SemanticLevel,
    SemanticSampleAnnotation,
    SemanticUnit,
    SemanticUnitKind,
    aggregate_semantic_wer,
    compute_semantic_wer,
)


def unit(
    ident: str,
    text: str,
    start: int,
    level: SemanticLevel,
    *,
    kind: SemanticUnitKind = "entity",
    role: str = "target",
    accepted: tuple[str, ...] = (),
    components: tuple[SemanticComponent, ...] = (),
) -> SemanticUnit:
    return SemanticUnit(
        id=ident,
        kind=kind,
        role=role,
        text=text,
        start_char=start,
        end_char=start + len(text),
        canonical_value=text,
        level=level,
        accepted_forms=accepted,
        components=components,
    )


def test_schema_identity_and_weighting() -> None:
    annotation = SemanticAnnotationSet(
        schema_version="0.1-draft",
        dataset_id="invented-set",
        dataset_identity="invented-v1",
        normalizer_version="2",
        samples=[
            SemanticSampleAnnotation(
                sample_id="sample", reference="hello", semantic_units=[unit("u", "hello", 0, 3)]
            )
        ],
    )
    assert annotation.dataset_identity == "invented-v1"
    assert LEVEL_WEIGHTS == {1: 0, 2: 1, 3: 3}


def test_surface_qualifier_and_core_errors_and_rows() -> None:
    reference = "please play quiet jazz"
    units = [
        unit("surface", "please", 0, 1),
        unit("qualifier", "quiet", 12, 2, kind="attribute"),
        unit("core", "jazz", 18, 3),
    ]
    result = compute_semantic_wer(reference, "kindly play loud blues", units)
    assert result.surface_errors == 1
    assert result.qualifier_errors == 1
    assert result.core_errors == 1
    assert result.error_weight == 4
    assert result.reference_weight == 4
    assert result.semantic_utterance_success is False

    sample_rows = {row.value_key: row for row in result.normalized_value_rows()}
    assert sample_rows["primary"].value == pytest.approx(100.0)
    assert sample_rows["semantic_successes"].value == 0
    assert sample_rows["utterances"].value == 1

    aggregate = aggregate_semantic_wer([result])
    rows = {row.value_key: row for row in aggregate.normalized_value_rows()}
    assert set(rows) == {
        "primary",
        "error_weight",
        "reference_weight",
        "core_errors",
        "core_units",
        "qualifier_errors",
        "qualifier_units",
        "surface_errors",
        "surface_units",
        "semantic_successes",
        "utterances",
    }
    assert rows["primary"].value == pytest.approx(100.0)
    assert rows["error_weight"].unit == "weight"
    assert rows["core_errors"].unit == "count"
    assert rows["primary"].value_role == "primary"
    assert rows["core_errors"].value_role == "component"


def test_accepted_forms_role_reversal_and_component_polarity() -> None:
    reference = "send it to north"
    destination = unit("destination", "north", 11, 3, role="destination", accepted=("northward",))
    result = compute_semantic_wer(reference, "send it north", [destination])
    assert result.unit_scores[0].outcome == "correct"
    reversed_result = compute_semantic_wer(reference, "send it to south", [destination])
    assert reversed_result.unit_scores[0].outcome == "substituted"

    ref = "keep sauce on side"
    components = (
        SemanticComponent(id="a", text="sauce", start_char=5, end_char=10),
        SemanticComponent(id="b", text="on", start_char=11, end_char=13),
        SemanticComponent(id="c", text="side", start_char=14, end_char=18),
    )
    placement = unit("placement", "sauce on side", 5, 3, components=components)
    polarity = compute_semantic_wer(ref, "keep sauce not on side", [placement])
    assert polarity.unit_scores[0].outcome == "substituted"


def test_identifier_forms_are_exact_and_reject_extra_characters() -> None:
    ref = "code A-7"
    identifier = unit("code", "A-7", 5, 3, kind="identifier")
    assert (
        compute_semantic_wer(ref, "code A dash 7", [identifier]).unit_scores[0].outcome == "correct"
    )
    assert (
        compute_semantic_wer(ref, "code A dash 7 please", [identifier]).unit_scores[0].outcome
        == "correct"
    )
    for hypothesis in ("code A dash 7 9", "code 9 A dash 7"):
        extra = compute_semantic_wer(ref, hypothesis, [identifier])
        assert extra.unit_scores[0].outcome == "substituted"


@pytest.mark.parametrize(
    ("hypothesis", "expected"),
    [
        ("say B dash 7, then A dash 8", "substituted"),
        ("code A dash 7 123", "substituted"),
        ("code A dash 7 123456", "substituted"),
        ("code 123 A dash 7", "substituted"),
        ("code A dash 7 z", "substituted"),
        ("code A-7ABCDE", "substituted"),
        ("code A dash 7 now", "correct"),
        ("code A dash 7", "correct"),
        ("code A-7", "correct"),
        ("code A 7", "substituted"),
    ],
)
def test_identifier_matching_is_alignment_local(hypothesis: str, expected: str) -> None:
    identifier = unit("code", "A-7", 5, 3, kind="identifier")
    result = compute_semantic_wer("code A-7", hypothesis, [identifier])
    assert result.unit_scores[0].outcome == expected


def test_exact_identifier_elsewhere_cannot_rescue_the_annotated_span() -> None:
    reference = "first A dash 7 then B dash 8"
    identifier = SemanticUnit(
        id="first",
        kind="identifier",
        role="first_code",
        text="A dash 7",
        start_char=6,
        end_char=14,
        canonical_value="A-7",
        level=3,
    )
    result = compute_semantic_wer(reference, "first A 7 then A dash 7", [identifier])
    assert result.unit_scores[0].outcome == "substituted"


def test_equal_length_substitutions_are_sliced_per_adjacent_unit() -> None:
    units = [
        unit("color", "red", 0, 3, accepted=("crimson",)),
        unit("fruit", "apples", 4, 3, accepted=("pears",)),
    ]
    result = compute_semantic_wer("red apples", "crimson pears", units)
    assert [score.outcome for score in result.unit_scores] == ["correct", "correct"]


@pytest.mark.parametrize(
    ("reference_value", "hypothesis_value", "canonical"),
    [
        ("twenty one", "21", "21"),
        ("one hundred", "100", "100"),
        ("fifty three", "53", "53"),
        ("one oh one", "101", "101"),
    ],
)
def test_contextual_number_normalization_preserves_identifier_identity(
    reference_value: str, hypothesis_value: str, canonical: str
) -> None:
    reference = f"value {reference_value}"
    identifier = unit("value", reference_value, 6, 3, kind="identifier")
    identifier = identifier.model_copy(update={"canonical_value": canonical})
    result = compute_semantic_wer(reference, f"value {hypothesis_value}", [identifier])
    assert result.unit_scores[0].outcome == "correct"


@pytest.mark.parametrize(
    ("hypothesis_value", "expected"),
    [
        ("A-7-B", "correct"),
        ("A dash 7 dash B", "correct"),
        ("A-7 B", "substituted"),
        ("A 7-B", "substituted"),
    ],
)
def test_repeated_identifier_dashes_are_preserved(hypothesis_value: str, expected: str) -> None:
    reference = "value A dash 7 dash B"
    identifier = unit("value", "A dash 7 dash B", 6, 3, kind="identifier")
    identifier = identifier.model_copy(update={"canonical_value": "A-7-B"})
    result = compute_semantic_wer(reference, f"value {hypothesis_value}", [identifier])
    assert result.unit_scores[0].outcome == expected


def test_spaced_literal_dash_and_edge_stars_are_required() -> None:
    spaced = "value A - 7"
    dash_unit = unit("value", "A - 7", 6, 3, kind="identifier")
    dash_unit = dash_unit.model_copy(update={"canonical_value": "A-7"})
    assert compute_semantic_wer(spaced, spaced, [dash_unit]).unit_scores[0].outcome == "correct"

    starred = "value star A dash 7 star"
    star_unit = unit("value", "star A dash 7 star", 6, 3, kind="identifier")
    star_unit = star_unit.model_copy(update={"canonical_value": "*A-7*"})
    assert (
        compute_semantic_wer(starred, "value *A-7*", [star_unit]).unit_scores[0].outcome
        == "correct"
    )
    assert (
        compute_semantic_wer(starred, "value * A-7 *", [star_unit]).unit_scores[0].outcome
        == "correct"
    )
    assert (
        compute_semantic_wer(starred, "value A-7*", [star_unit]).unit_scores[0].outcome
        == "substituted"
    )
    assert (
        compute_semantic_wer(starred, "value *A-7", [star_unit]).unit_scores[0].outcome
        == "substituted"
    )


@pytest.mark.parametrize(
    ("canonical", "spoken", "missing"),
    [
        ("-A", "dash A", "A"),
        ("A-", "A dash", "A"),
        ("-A-", "dash A dash", "A"),
        ("A--B", "A dash dash B", "A-B"),
    ],
)
def test_edge_and_consecutive_identifier_dashes_are_required(
    canonical: str, spoken: str, missing: str
) -> None:
    reference = f"value {spoken}"
    identifier = unit("value", spoken, 6, 3, kind="identifier")
    identifier = identifier.model_copy(update={"canonical_value": canonical})

    assert (
        compute_semantic_wer(reference, f"value {canonical}", [identifier]).unit_scores[0].outcome
        == "correct"
    )
    assert (
        compute_semantic_wer(reference, f"value {spoken}", [identifier]).unit_scores[0].outcome
        == "correct"
    )
    assert (
        compute_semantic_wer(reference, f"value {missing}", [identifier]).unit_scores[0].outcome
        == "substituted"
    )


def test_repeated_adjacent_tokens_are_not_collapsed() -> None:
    units = [
        unit("first", "red", 0, 3, accepted=("crimson",)),
        unit("second", "red", 4, 3, accepted=("crimson",)),
    ]
    result = compute_semantic_wer("red red", "crimson crimson", units)
    assert [score.outcome for score in result.unit_scores] == ["correct", "correct"]


def test_compact_identifier_preserves_required_symbols() -> None:
    reference = "code A dash 7 star B"
    identifier = SemanticUnit(
        id="code",
        kind="identifier",
        role="target",
        text="A dash 7 star B",
        start_char=5,
        end_char=len(reference),
        canonical_value="A-7*B",
        level=3,
    )

    exact = compute_semantic_wer(reference, "code A-7*B", [identifier])
    missing_symbol = compute_semantic_wer(reference, "code A-7 B", [identifier])
    assert exact.unit_scores[0].outcome == "correct"
    assert missing_symbol.unit_scores[0].outcome == "substituted"


def test_deleted_qualifier_is_weighted_without_failing_core_success() -> None:
    reference = "play quiet jazz"
    units = [
        unit("qualifier", "quiet", 5, 2, kind="attribute"),
        unit("core", "jazz", 11, 3),
    ]
    result = compute_semantic_wer(reference, "play jazz", units)

    assert result.unit_scores[0].outcome == "deleted"
    assert result.error_weight == 1
    assert result.reference_weight == 4
    assert result.semantic_unit_error_rate == pytest.approx(0.25)
    assert result.semantic_utterance_success


def test_surface_only_and_empty_aggregates_use_zero_denominators() -> None:
    surface = compute_semantic_wer(
        "please continue",
        "kindly continue",
        [unit("surface", "please", 0, 1)],
    )
    assert surface.semantic_unit_error_rate == 0.0
    assert surface.reference_weight == 0

    aggregate = aggregate_semantic_wer([])
    assert aggregate.semantic_unit_error_rate == 0.0
    assert aggregate.semantic_utterance_success_rate == 0.0
    assert aggregate.samples == 0
    assert {row.value_key: row.value for row in aggregate.normalized_value_rows()}["primary"] == 0


def test_pooling_and_json_round_trip() -> None:
    ref = "play jazz"
    units = [unit("core", "jazz", 5, 3)]
    good = compute_semantic_wer(ref, ref, units)
    bad = compute_semantic_wer(ref, "play blues", units)
    pooled = aggregate_semantic_wer([good, bad])
    assert pooled.error_weight == 3
    assert pooled.reference_weight == 6
    assert SemanticUnit.model_validate_json(units[0].model_dump_json()) == units[0]
    assert pooled.model_validate_json(pooled.model_dump_json()) == pooled


def test_annotation_set_rejects_duplicate_samples() -> None:
    sample = SemanticSampleAnnotation(
        sample_id="duplicate",
        reference="play jazz",
        semantic_units=[unit("core", "jazz", 5, 3)],
    )
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        SemanticAnnotationSet(
            schema_version="0.1-draft",
            dataset_id="invented-set",
            dataset_identity="invented-v1",
            normalizer_version="2",
            samples=[sample, sample],
        )


def test_annotation_rejects_text_that_does_not_match_reference_span() -> None:
    with pytest.raises(ValueError, match="text does not match"):
        SemanticSampleAnnotation(
            sample_id="mismatch",
            reference="play jazz",
            semantic_units=[unit("core", "blue", 5, 3)],
        )


@pytest.mark.parametrize(
    "units",
    [
        [unit("duplicate", "play", 0, 3), unit("duplicate", "jazz", 5, 3)],
        [unit("overlap", "play jazz", 0, 3), unit("nested", "jazz", 5, 3)],
    ],
)
def test_invalid_duplicate_or_overlapping_spans_rejected(units: list[SemanticUnit]) -> None:
    with pytest.raises(ValueError):
        compute_semantic_wer("play jazz", "play jazz", units)
