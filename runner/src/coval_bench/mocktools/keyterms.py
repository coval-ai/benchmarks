# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain vocabulary, read back out of the fixtures.

``stack.json`` pins Deepgram nova-3 and recovers dental vocabulary through
keyterm prompting rather than a medical model, because no hosted platform offers
one. The keyterms are generated from the seeded world instead of hand-kept: the
words the agent has to hear correctly are exactly the patient names, providers
and visit types the tools will say back to it, and a hand-maintained list would
drift from the fixtures the first time a scenario was added.

Only declared keys are harvested. Walking every string in the responses would
sweep up confirmation numbers and free text, and a keyterm list padded with
noise costs recognition accuracy on the terms that matter.
"""

from __future__ import annotations

from coval_bench.mocktools.dispatch import ToolSpec
from coval_bench.mocktools.fixtures import MockFixtures

# Response fields that hold vocabulary rather than data. Extend this rather than
# widening the walk: every added key should be a word a caller might say aloud.
KEYTERM_KEYS = frozenset(
    {
        "patient_name",
        "provider",
        "provider_name",
        "dentist",
        "clinic",
        "appointment_type",
    }
)


def _spoken(term: str) -> str:
    """A schema token as a caller would say it: ``root_canal_consult`` is three words."""
    return term.strip().replace("_", " ")


def _harvest(node: object, found: set[str]) -> None:
    """Collect declared-key string values anywhere in a response tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in KEYTERM_KEYS and isinstance(value, str) and value.strip():
                found.add(value.strip())
            else:
                _harvest(value, found)
    elif isinstance(node, list):
        for item in node:
            _harvest(item, found)


def extract_keyterms(specs: dict[str, ToolSpec], fixtures: MockFixtures) -> tuple[str, ...]:
    """The suite's keyterm list: fixture vocabulary plus every contract enum value.

    Returns them sorted, so the list a platform is configured with is stable
    across runs and a diff means the fixtures actually changed.
    """
    found: set[str] = set()
    for spec in specs.values():
        for name, values in spec.enums.items():
            # Enums are filtered by the same declared keys as responses. An
            # `appointment_type` is said out loud; a `reason` is an internal token,
            # and prompting the STT with "tool_failure" only costs accuracy on the
            # terms a caller actually says.
            if name not in KEYTERM_KEYS:
                continue
            found.update(_spoken(value) for value in values if value.strip())
    for fixture in fixtures.tools.values():
        for seed in (*fixture.seeds, fixture.fallback):
            _harvest(seed.response, found)
    return tuple(sorted(found))
