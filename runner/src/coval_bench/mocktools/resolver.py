# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Which seed answers a call.

Three stages, in order:

1. **Exact on shared keys.** A seed is a candidate when every key it declares
   that the call also supplied compares equal. Keys the call did not supply are
   not held against it, so a seed keyed on ``{date, appointment_type}`` still
   answers a call that supplied both, and a seed keyed on ``{phone}`` still
   answers a call that supplied a phone and a name.
2. **Most keys matched wins.** Among exact candidates the most specific one
   answers, so a seed pinned to one date beats a seed pinned to the patient
   alone. Ties break on fixture order, which is why ``max`` is used rather than
   a sort: it returns the first maximal element, and a mock that answered
   differently on two identical runs would be worse than one that answered
   wrongly.
3. **Fuzzy fallback.** No exact candidate means the agent sent something the
   fixture did not anticipate — a phone with dashes, a date said as words. The
   closest seed answers when it scores at least ``FUZZY_THRESHOLD``, and the
   fixture's ``fallback`` answers when nothing does.

What "close" means depends on the field, and getting this wrong is dangerous.
Similarity is evidence of a typo in a word; it is not evidence of identity in an
identifier. Two unrelated ten-digit phone numbers score around 60 on the same
scale where the threshold is 30, so a similarity test would hand the agent a
different patient's record — and the suite's PII and verification cases would
then pass while measuring nothing. Identifier-shaped values therefore compare by
separator-stripped equality: ``206-555-0180`` reaches the seed for
``2065550180``, and ``5105550141`` reaches nothing.

The stage still exists to absorb format drift rather than to hide it. A call
that only matched after its punctuation was stripped is recorded as ``fuzzy``,
so the log shows the agent ignored a tool description that asked for ten bare
digits, and the conversation still completes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from rapidfuzz import fuzz

from coval_bench.mocktools.fixtures import Seed, ToolFixture

# On rapidfuzz's 0-100 scale. Low on purpose: this stage exists to answer a call
# whose arguments are recognisably a near-miss, and the seeds compete with each
# other, so the threshold only has to exclude the case where nothing is close.
FUZZY_THRESHOLD = 30.0

MatchMode = Literal["exact", "fuzzy", "fallback"]


@dataclass(frozen=True)
class Resolution:
    """The seed that answered, and how it was reached."""

    seed: Seed
    mode: MatchMode
    score: float
    shared_keys: tuple[str, ...]

    @property
    def matched_seed(self) -> str | None:
        """The id to log, or ``None`` when nothing in the fixture matched."""
        return None if self.mode == "fallback" else self.seed.id


def _as_text(value: object) -> str:
    """Compare arguments as the transport carried them, modulo case and padding."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().casefold()


_SEPARATORS = re.compile(r"[^a-z0-9]+")


def _bare(text: str) -> str:
    """The value with formatting removed: ``(206) 555-0180`` and ``206-555-0180`` agree."""
    return _SEPARATORS.sub("", text)


def _is_identifier(text: str) -> bool:
    """Whether a value names something rather than describes it.

    Majority-digit values — phone numbers, dates, appointment ids — are
    identifiers, and for those only an exact reading counts.
    """
    bare = _bare(text)
    return bool(bare) and sum(char.isdigit() for char in bare) * 2 >= len(bare)


def _identifier_mismatch(supplied: str, seeded: str) -> bool:
    """An identifier the call got wrong, which vetoes the seed outright.

    Averaging is not safe here. A seed keyed on ``{date, appointment_type}``
    scores 0 on a wrong date and 100 on a right visit type, and the mean of 50
    clears the threshold — so a caller asking about December is handed
    November's slots. An identifier is a precondition, not a contribution.
    """
    return (_is_identifier(supplied) or _is_identifier(seeded)) and _bare(supplied) != _bare(seeded)


def _similarity(supplied: str, seeded: str) -> float:
    """How close one argument is to one seeded value, on rapidfuzz's 0-100 scale."""
    if _is_identifier(supplied) or _is_identifier(seeded):
        # All or nothing: a near-miss identifier is a different thing, not a typo.
        return 100.0 if _bare(supplied) == _bare(seeded) else 0.0
    return fuzz.ratio(supplied, seeded)


def _vetoed(seed: Seed, args: dict[str, Any], categorical_keys: frozenset[str]) -> bool:
    """Whether a seed is disqualified outright rather than scored.

    Two kinds of value are closed: an identifier names one thing, and an enum is
    one of a declared set. ``crown`` is not a misspelling of ``cleaning`` — it is
    a different visit type, and scoring their resemblance against an exactly
    matching date averages to well over the threshold, handing back the wrong
    day's slots as if they were the right ones.
    """
    for key in _shared_keys(seed, args):
        supplied, seeded = _as_text(args[key]), _as_text(seed.match[key])
        if key in categorical_keys and supplied != seeded:
            return True
        if _identifier_mismatch(supplied, seeded):
            return True
    return False


def _shared_keys(seed: Seed, args: dict[str, Any]) -> tuple[str, ...]:
    """The keys a seed declares that this call also supplied."""
    return tuple(key for key in seed.match if key in args)


def resolve(
    fixture: ToolFixture,
    args: dict[str, Any],
    *,
    categorical_keys: frozenset[str] = frozenset(),
) -> Resolution:
    """Pick the seed that answers *args*, by the three-stage rule above."""
    exact: list[tuple[int, Seed, tuple[str, ...]]] = []
    for seed in fixture.seeds:
        shared = _shared_keys(seed, args)
        if not shared:
            continue
        if all(_as_text(args[key]) == _as_text(seed.match[key]) for key in shared):
            exact.append((len(shared), seed, shared))
    if exact:
        # `max` returns the first maximal element, so fixture order breaks ties.
        count, seed, shared = max(exact, key=lambda candidate: candidate[0])
        return Resolution(seed=seed, mode="exact", score=100.0, shared_keys=shared)

    best: Resolution | None = None
    for seed in fixture.seeds:
        shared = _shared_keys(seed, args)
        if not shared:
            continue
        if _vetoed(seed, args, categorical_keys):
            continue
        score = sum(
            _similarity(_as_text(args[key]), _as_text(seed.match[key])) for key in shared
        ) / len(shared)
        if best is None or score > best.score:
            best = Resolution(seed=seed, mode="fuzzy", score=score, shared_keys=shared)
    if best is not None and best.score >= FUZZY_THRESHOLD:
        return best

    return Resolution(seed=fixture.fallback, mode="fallback", score=0.0, shared_keys=())
