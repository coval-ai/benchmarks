# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, candidate-blind stratification for in-domain manual review."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class STTSelectionItem(Protocol):
    path: str
    sha256: str
    transcript: str
    duration_sec: float


@dataclass(frozen=True, slots=True, order=True)
class ClipStratum:
    duration: str
    transcript_length: str
    style: str


_COMMAND_VERBS = frozenset(
    {
        "add",
        "adjust",
        "announce",
        "book",
        "call",
        "cancel",
        "change",
        "check",
        "close",
        "create",
        "delete",
        "find",
        "list",
        "lock",
        "navigate",
        "open",
        "order",
        "play",
        "remind",
        "remove",
        "schedule",
        "search",
        "send",
        "set",
        "show",
        "start",
        "stop",
        "turn",
        "update",
    }
)


def clip_stratum(item: STTSelectionItem) -> ClipStratum:
    """Classify with fixed rules that are frozen before candidate outputs are viewed."""
    word_count = len(item.transcript.split())
    duration = "short" if item.duration_sec < 4 else "medium" if item.duration_sec < 8 else "long"
    transcript_length = "short" if word_count < 10 else "medium" if word_count < 25 else "long"
    first_word = item.transcript.lstrip().split(maxsplit=1)[0].casefold().strip("'\".,?!")
    style = "command_like" if first_word in _COMMAND_VERBS else "conversational_or_question"
    return ClipStratum(
        duration=duration,
        transcript_length=transcript_length,
        style=style,
    )


def select_stratified_stt_items[Item: STTSelectionItem](
    items: Sequence[Item],
    *,
    sample_size: int,
    seed: str,
) -> tuple[Item, ...]:
    """Round-robin deterministic strata, with hash ordering inside each stratum."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(items):
        raise ValueError("sample_size cannot exceed the number of available items")
    grouped: dict[ClipStratum, list[Item]] = defaultdict(list)
    for item in items:
        grouped[clip_stratum(item)].append(item)
    queues: dict[ClipStratum, deque[Item]] = {}
    for stratum, members in grouped.items():
        ordered = sorted(
            members,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item.sha256}:{item.path}".encode()
            ).hexdigest(),
        )
        queues[stratum] = deque(ordered)
    selected: list[Item] = []
    while len(selected) < sample_size:
        made_progress = False
        for stratum in sorted(queues):
            if queues[stratum] and len(selected) < sample_size:
                selected.append(queues[stratum].popleft())
                made_progress = True
        if not made_progress:
            raise RuntimeError("stratified sampling exhausted items unexpectedly")
    return tuple(selected)
