# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from coval_bench.metrics import compute_cer
from coval_bench.registries import Metric
from coval_bench.registries.metrics import MetricValueRole, validate_metric_values


@pytest.mark.parametrize(
    ("ref", "hyp", "errors", "reference_chars"),
    [
        ("hello world", "Hello, world.", 0, 11),
        ("bennett", "bennet", 1, 7),
        ("turnaround today", "turn around today", 1, 16),
        ("hello", "", 5, 5),
        ("", "", 0, 0),
        ("", "hi", 2, 0),
    ],
)
def test_compute_cer(ref: str, hyp: str, errors: int, reference_chars: int) -> None:
    result = compute_cer(ref, hyp)
    assert (result.errors, result.reference_chars) == (errors, reference_chars)
    expected = errors / reference_chars if reference_chars else float(errors)
    assert result.cer == pytest.approx(expected)
    assert result.cer_percentage == pytest.approx(expected * 100)


def test_character_counts_are_all_or_none() -> None:
    primary = ("primary", "percent", 14.3, MetricValueRole.PRIMARY)
    errors = ("error_chars", "count", 1.0, MetricValueRole.COMPONENT)
    reference = ("reference_chars", "count", 7.0, MetricValueRole.COMPONENT)
    validate_metric_values(Metric.CHARACTER_ERROR_RATE, "v1", (primary,))
    validate_metric_values(Metric.CHARACTER_ERROR_RATE, "v1", (primary, errors, reference))
    with pytest.raises(ValueError, match="incomplete"):
        validate_metric_values(Metric.CHARACTER_ERROR_RATE, "v1", (primary, errors))
