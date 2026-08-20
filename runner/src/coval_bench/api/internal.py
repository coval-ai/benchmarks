# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who may see hidden models.

Hidden models (``running`` without ``shown`` in ``model_state`` — the
pre-launch embargo) run on the normal schedule but every data endpoint strips
them unless the caller proves it may see them. The one proof is a bearer Clerk
session token (see ``clerk.py``): a coval.dev email sees everything, and a
mapped provider org sees what its entry names. The grants live in settings, so
a request can never widen its own view.

An absent or unknown proof yields the public view — the endpoints stay public
either way, so there is nothing to 404 — and says so in ``X-EA-Token-Status``.
"""

from __future__ import annotations

from fastapi import Depends, Header, Response

from coval_bench.api import clerk
from coval_bench.api.deps import get_model_states, get_settings
from coval_bench.config import Settings
from coval_bench.db.model_state import ModelKey, ModelState
from coval_bench.registries import MODEL_REGISTRY

# Which proof the response honoured: accepted, unknown, or absent.
EA_STATUS_HEADER = "X-EA-Token-Status"

# The one proof header a response's content may depend on.
VARY_HEADERS = "Authorization"


def never_shared(response: Response) -> None:
    """Mark a response as belonging to this caller alone.

    Any response whose content depends on the presented proof needs this, errors
    included: the same URL is a 404 for the public and a redirect for a partner,
    so a shared cache must never hand one caller's answer to another.
    """
    response.headers.append("Vary", VARY_HEADERS)
    response.headers["Cache-Control"] = "private, no-store"


# Board keys an embargoed model used to be published under, mapped to the key it
# uses now. Sample manifests and result rows written before a rename still carry
# the old string and the embargo matches on what is stored, so a retired key stays
# embargoed -- dropping an entry here re-exposes every artefact published under
# that name. Retire one only once nothing stored can still name it.
_RETIRED_BOARD_KEYS: dict[tuple[str, str], tuple[str, str]] = {
    ("xai", "grok-realtime"): ("xai", "grok-voice-think-fast-1.0"),
}


def embargoed_pairs(states: dict[ModelKey, ModelState]) -> frozenset[tuple[str, str]]:
    """Every ``(provider, model)`` pair currently under embargo (Hidden state)."""
    return frozenset(
        (provider, model)
        for (_, provider, model), state in states.items()
        if state.running and not state.shown
    ) | frozenset(_RETIRED_BOARD_KEYS)


def with_retired_keys(allowed: frozenset[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    """Extend an allowlist to the retired keys of the models it already names.

    An allowlist names models, not the strings they were once published under, so
    clearing a caller for a model clears them for its history too. Without this a
    rename would silently narrow every partner's view to artefacts written after
    it, and each operator would have to know to list both keys by hand.
    """
    return allowed | frozenset(
        retired for retired, current in _RETIRED_BOARD_KEYS.items() if current in allowed
    )


def all_registered_pairs() -> frozenset[tuple[str, str]]:
    """Every pair the registry knows, embargoed or not, plus the retired keys.

    The universe an exclusive org's view is subtracted from. A pair that was never
    registered cannot be subtracted, so it stays visible.
    """
    return frozenset((m.provider, m.model) for m in MODEL_REGISTRY) | frozenset(_RETIRED_BOARD_KEYS)


async def hidden_early_access(
    response: Response,
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
    states: dict[ModelKey, ModelState] = Depends(get_model_states),
) -> frozenset[tuple[str, str]]:
    """The pairs this caller's responses must not contain.

    Cache headers are set here rather than per route, so no endpoint can serve
    embargoed rows without them.
    """
    # Appended, not assigned: SelectiveGZipMiddleware adds Accept-Encoding after
    # the route returns, and assignment here would be overwritten.
    response.headers.append("Vary", VARY_HEADERS)

    embargoed = embargoed_pairs(states)
    if authorization is None or settings.clerk_issuer is None:
        response.headers[EA_STATUS_HEADER] = "absent"
        return embargoed

    # Exclusive orgs are resolved first and never widened: nothing may add to a
    # view whose whole point is what it leaves out.
    universe = all_registered_pairs()
    only = clerk.exclusive_pairs(authorization, settings, universe)
    if only is not None:
        response.headers[EA_STATUS_HEADER] = "accepted"
        response.headers["Cache-Control"] = "private, no-store"
        return universe - with_retired_keys(only)

    bearer = clerk.allowed_pairs(authorization, settings, embargoed)
    if bearer is None:
        response.headers[EA_STATUS_HEADER] = "unknown"
        return embargoed

    response.headers[EA_STATUS_HEADER] = "accepted"
    response.headers["Cache-Control"] = "private, no-store"
    return embargoed - with_retired_keys(bearer)
