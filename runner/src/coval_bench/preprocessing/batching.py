# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, integer-millisecond batching for preprocessing workers."""

from __future__ import annotations

from collections.abc import Callable, Iterable

DEFAULT_MAX_BATCH_CLIPS = 32
DEFAULT_MAX_BATCH_DURATION_MS = 300_000


def batch_by_duration[Item](
    items: Iterable[Item],
    *,
    duration_ms_of: Callable[[Item], int],
    max_clips: int = DEFAULT_MAX_BATCH_CLIPS,
    max_duration_ms: int = DEFAULT_MAX_BATCH_DURATION_MS,
) -> tuple[tuple[Item, ...], ...]:
    """Batch in input order under clip-count and total-duration bounds.

    A clip longer than ``max_duration_ms`` is retained as a singleton. Integer
    milliseconds make exact-boundary behavior independent of float arithmetic.
    """
    if isinstance(max_clips, bool) or not isinstance(max_clips, int) or max_clips < 1:
        raise ValueError("max_clips must be a positive integer")
    if (
        isinstance(max_duration_ms, bool)
        or not isinstance(max_duration_ms, int)
        or max_duration_ms < 1
    ):
        raise ValueError("max_duration_ms must be a positive integer")

    batches: list[tuple[Item, ...]] = []
    current: list[Item] = []
    current_duration_ms = 0

    for item in items:
        duration_ms = duration_ms_of(item)
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 1:
            raise ValueError("item duration must be a positive integer number of milliseconds")

        if duration_ms > max_duration_ms:
            if current:
                batches.append(tuple(current))
                current = []
                current_duration_ms = 0
            batches.append((item,))
            continue

        if current and (
            len(current) >= max_clips or current_duration_ms + duration_ms > max_duration_ms
        ):
            batches.append(tuple(current))
            current = []
            current_duration_ms = 0

        current.append(item)
        current_duration_ms += duration_ms

    if current:
        batches.append(tuple(current))
    return tuple(batches)
