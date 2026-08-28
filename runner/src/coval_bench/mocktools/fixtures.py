# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The seeded world the mock tools answer from.

Fixtures live at ``contracts/<suite>/_private/mock-tools.json`` and are never
committed: they are the answer key. A scenario that says "3:00 PM is
unavailable" only grades if the agent cannot read that fact anywhere but the
tool, so the file sits behind the same ``_private/`` rule as the test cases and
is hashed into the published contract sha without being published itself.

Shape::

    {
      "tools": {
        "lookup_patient": {
          "seeds": [
            {"id": "rina_das", "match": {"phone": "4085550119"},
             "response": {"found": true, "patient_id": "P-1041"}}
          ],
          "fallback": {"id": "no_match", "response": {"found": false}}
        }
      }
    }

``match`` holds the argument values a seed answers to. ``fallback`` is what the
tool returns when nothing matches, and every tool must declare one: a mock with
no answer for an unseeded input would fail the call, and a failed call is a
measurement of our fixtures rather than of the agent.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, field_validator

from coval_bench.contracts import AnnotatedModel


class Seed(AnnotatedModel):
    """One seeded answer: the arguments it responds to, and what it returns."""

    id: str = Field(min_length=1)
    # Values are compared as text. Numbers and dates are quoted in the fixture so
    # that what the resolver compares is what the transport actually carried.
    match: dict[str, str] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    # A seed may model a broken backend. The agent's recovery from a 500 is part
    # of what the suite grades, so the failure has to be seedable like any answer.
    http_status: int = Field(default=200, ge=200, le=599)


class ToolFixture(AnnotatedModel):
    """Every seeded answer for one tool, plus the answer of last resort."""

    seeds: tuple[Seed, ...] = ()
    fallback: Seed

    @field_validator("fallback")
    @classmethod
    def _fallback_matches_nothing(cls, fallback: Seed) -> Seed:
        """A fallback answers when matching failed, so its own ``match`` is never read.

        Allowing keys there would let an author write a condition that looks
        load-bearing and silently is not.
        """
        if fallback.match:
            raise ValueError(
                f"fallback seed {fallback.id!r} declares match keys "
                f"({', '.join(sorted(fallback.match))}); a fallback is not matched on"
            )
        return fallback

    @field_validator("seeds")
    @classmethod
    def _unique_ids(cls, seeds: tuple[Seed, ...]) -> tuple[Seed, ...]:
        """Seed ids land in ``mock_tool_calls.matched_seed``; duplicates make it useless."""
        ids = [seed.id for seed in seeds]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate seed ids: {', '.join(duplicates)}")
        return seeds


class MockFixtures(AnnotatedModel):
    """The whole seeded world, keyed by tool name."""

    tools: dict[str, ToolFixture]


def parse_fixtures(raw: str | bytes) -> MockFixtures:
    """Parse and validate a ``mock-tools.json`` document."""
    return MockFixtures.model_validate(json.loads(raw))
