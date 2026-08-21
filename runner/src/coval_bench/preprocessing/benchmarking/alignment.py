# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic unit-cost sequence alignment for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AlignmentOperation(StrEnum):
    MATCH = "match"
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"


@dataclass(frozen=True, slots=True)
class AlignmentStep[Item]:
    operation: AlignmentOperation
    reference_index: int | None
    hypothesis_index: int | None
    reference: Item | None
    hypothesis: Item | None


def align_sequences[SequenceItem](
    reference: tuple[SequenceItem, ...], hypothesis: tuple[SequenceItem, ...]
) -> tuple[AlignmentStep[SequenceItem], ...]:
    """Return a stable Levenshtein alignment, preferring diagonal ties."""
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        costs[row][0] = row
    for column in range(columns):
        costs[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = 0 if reference[row - 1] == hypothesis[column - 1] else 1
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + substitution_cost,
            )

    steps: list[AlignmentStep[SequenceItem]] = []
    row = len(reference)
    column = len(hypothesis)
    while row or column:
        if row and column:
            is_match = reference[row - 1] == hypothesis[column - 1]
            diagonal_cost = costs[row - 1][column - 1] + (0 if is_match else 1)
            if costs[row][column] == diagonal_cost:
                steps.append(
                    AlignmentStep(
                        operation=(
                            AlignmentOperation.MATCH
                            if is_match
                            else AlignmentOperation.SUBSTITUTION
                        ),
                        reference_index=row - 1,
                        hypothesis_index=column - 1,
                        reference=reference[row - 1],
                        hypothesis=hypothesis[column - 1],
                    )
                )
                row -= 1
                column -= 1
                continue
        if row and costs[row][column] == costs[row - 1][column] + 1:
            steps.append(
                AlignmentStep(
                    operation=AlignmentOperation.DELETION,
                    reference_index=row - 1,
                    hypothesis_index=None,
                    reference=reference[row - 1],
                    hypothesis=None,
                )
            )
            row -= 1
            continue
        steps.append(
            AlignmentStep(
                operation=AlignmentOperation.INSERTION,
                reference_index=None,
                hypothesis_index=column - 1,
                reference=None,
                hypothesis=hypothesis[column - 1],
            )
        )
        column -= 1

    steps.reverse()
    return tuple(steps)
