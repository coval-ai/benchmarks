# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic domain models for the Voice Arena persistence layer.

Persistence-layer models only (mirrors ``db/models.py``). The API layer has its
own schemas; do not couple them to these.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

__all__ = ["Battle", "Vote", "VoteOutcome", "VoterType"]


class VoteOutcome(StrEnum):
    """Outcome of a single A/B battle vote."""

    A_WIN = "A_WIN"
    B_WIN = "B_WIN"
    TIE = "TIE"


class VoterType(StrEnum):
    """Who cast the vote. ``external`` is reserved for future public voting."""

    LABELER = "labeler"
    EXTERNAL = "external"


class Battle(BaseModel):
    """Domain model for a row in ``arena.battles`` — one A vs B matchup."""

    id: UUID | None = None  # set by DB (gen_random_uuid)
    provider_a: str
    model_a: str
    provider_b: str
    model_b: str
    domain: str | None = None
    prompt_text: str
    audio_a_url: str
    audio_b_url: str
    created_at: datetime | None = None  # set by DB default (now())


class Vote(BaseModel):
    """Domain model for a row in ``arena.votes`` — one human judgment."""

    id: UUID | None = None  # set by DB (gen_random_uuid)
    battle_id: UUID
    outcome: VoteOutcome
    voter_type: VoterType
    voter_id: str
    created_at: datetime | None = None  # set by DB default (now())
    updated_at: datetime | None = None  # maintained by the BEFORE UPDATE trigger
