# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""The shape of a benchmarked model: identity, run config, and state.

One ``RegisteredModel`` per ``(benchmark, provider, model)``. The rows live in
``benchmarks_v2.models`` and are read through ``db.registry_store.fetch_models``;
the orchestrator runs every ``collected`` entry and the API serves only the
``published`` ones to the public.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel

from coval_bench.registries.benchmarks import Benchmark


class Source(StrEnum):
    """Where the benchmarked endpoint lives: the creator's own API or an inference host."""

    OFFICIAL_API = "official-api"
    SHARED_INFERENCE = "shared-inference"
    DEDICATED_INFERENCE = "dedicated-inference"


class Licensing(StrEnum):
    """Whether the model's weights are openly available."""

    PROPRIETARY = "proprietary"
    OPEN_WEIGHT = "open-weight"


class Gender(StrEnum):
    """Perceived gender of a voice — an editorial label, not ground truth.

    Provider metadata is too uneven to encode provenance: Hume tags every stock
    voice, OpenAI publishes none. Four providers here are API-confirmed
    (cartesia, hume, xai, lmnt); the rest are our own listening.
    """

    FEMALE = "female"
    MALE = "male"


# What the site can draw: six hex digits, any case on the way in, lowercase at rest.
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def normalize_hex_color(value: str | None) -> str | None:
    """A ``#rrggbb`` color lowercased, ``None`` passed through; anything else is a ValueError."""
    if value is None:
        return None
    candidate = value.strip()
    if not _HEX_COLOR.match(candidate):
        raise ValueError("a color is six hex digits behind a #, like #1db098")
    return candidate.lower()


# A model's series color as every writer accepts it: null, or #rrggbb in any case,
# stored lowercase. One type so no body that carries a color can skip the rule.
HexColor = Annotated[str | None, AfterValidator(normalize_hex_color)]


class Voice(BaseModel, frozen=True, extra="forbid"):
    """One speaker, as the provider addresses it.

    ``id`` goes on the wire and is often an opaque UUID, so ``name`` keeps
    registry diffs readable. ``accent`` is recorded where known, not controlled.

    Nothing pairs on ``gender`` yet: the arena synthesizes with the scalar
    ``RegisteredModel.voice``, which has no gender and for most models isn't
    even a pool member. Battles today are cross-gender by construction.
    """

    id: str
    gender: Gender
    name: str | None = None
    accent: str | None = None


class RegisteredModel(BaseModel, frozen=True, extra="forbid"):
    """A single benchmarked model: identity, display metadata, run config."""

    benchmark: Benchmark
    provider: str
    model: str
    voice: str | None = None  # TTS only
    # TTS only: balanced voice pool, one :class:`Voice` per gender. Each run
    # splits its samples evenly across the pool; ``voice`` is the fallback for
    # models without one. Gender lives on the ``Voice``, not in tuple order —
    # see :class:`Voice` for why nothing pairs on it yet.
    voices: tuple[Voice, ...] = ()
    creator: str | None = None  # who makes the model; None means same as provider
    tags: tuple[str, ...] = ()
    source: Source = Source.OFFICIAL_API
    licensing: Licensing = Licensing.PROPRIETARY
    on_prem: bool = False  # provider offers on-prem/customer-infra deployment
    # Coarse server location of the endpoint we benchmark. Unset means
    # unknown. Geo- or globally routed endpoints record where our runner's
    # traffic lands.
    region: Literal["us", "eu", "asia"] | None = None
    # Do we spend money measuring it: the orchestrator schedules runs for it and
    # the arena may pair it.
    collected: bool
    # Are its results public. An unpublished model's results are served only to
    # the org its grant names, never to everyone.
    published: bool
    arena_enabled: bool = True  # in the arena roster? independent of `collected`
    # The series color the site draws the model in, as lowercase ``#rrggbb``.
    # None means the site picks one from its built-in palette.
    color: HexColor = None
