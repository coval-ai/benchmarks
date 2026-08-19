# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who may see early-access models.

``EARLY_ACCESS`` models run on the normal schedule but are under embargo: every
data endpoint strips them unless the caller proves it may see them. The
benchmarking team presents ``X-Internal-Key`` and sees everything; a partner
presents ``X-EA-Token``, which the server resolves to an allowlist of the models
that token may see. The token is opaque and the allowlist lives in settings, so a
request can never widen its own view. A signed-in provider org presents a
bearer Clerk session token instead (see ``clerk.py``).

An absent or unknown proof yields the public view — the endpoints stay public
either way, so there is nothing to 404 — and says so in ``X-EA-Token-Status``.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
from collections.abc import Mapping

import structlog
from fastapi import Depends, Header, Response

from coval_bench.api import clerk
from coval_bench.api.deps import get_settings
from coval_bench.config import Settings
from coval_bench.registries import MODEL_REGISTRY, ModelStatus

logger = structlog.get_logger("coval_bench.api.internal")

# Which proof the response honoured: internal, accepted, unknown, or absent.
EA_STATUS_HEADER = "X-EA-Token-Status"

# Every proof header. Listing a subset is worse than listing none: a cached
# internal response carries no X-EA-Token, so it would match a public request.
VARY_HEADERS = "X-Internal-Key, X-EA-Token, Authorization"


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


def embargoed_pairs() -> frozenset[tuple[str, str]]:
    """Every ``(provider, model)`` pair currently under embargo."""
    return frozenset(
        (m.provider, m.model) for m in MODEL_REGISTRY if m.status is ModelStatus.EARLY_ACCESS
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


def hidden_models() -> frozenset[tuple[str, str]]:
    """The pairs public API responses must not contain."""
    return embargoed_pairs()


@functools.lru_cache(maxsize=1)
def _parse_allowlists(raw: str) -> Mapping[str, frozenset[tuple[str, str]]]:
    """Parse the ``{token: ["provider/model", ...]}`` settings blob.

    Split on the first ``/``, so a model id containing one still parses. A
    malformed blob yields an empty table: a bad deploy falls back to the public
    view rather than taking the data endpoints down.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("early_access_tokens_unparsable")
        return {}
    if not isinstance(parsed, dict):
        logger.error("early_access_tokens_not_an_object")
        return {}

    table: dict[str, frozenset[tuple[str, str]]] = {}
    for token, entries in parsed.items():
        if not isinstance(token, str) or not isinstance(entries, list):
            logger.error("early_access_token_entry_malformed")
            continue
        pairs = set()
        for entry in entries:
            provider, sep, model = entry.partition("/") if isinstance(entry, str) else ("", "", "")
            if not sep or not provider or not model:
                logger.error("early_access_entry_not_provider_slash_model")
                continue
            pairs.add((provider, model))
        table[token] = frozenset(pairs)
    return table


def _allowlists(settings: Settings) -> Mapping[str, frozenset[tuple[str, str]]]:
    blob = settings.early_access_tokens
    return {} if blob is None else _parse_allowlists(blob.get_secret_value())


def _allowed_for(
    token: str, table: Mapping[str, frozenset[tuple[str, str]]]
) -> frozenset[tuple[str, str]] | None:
    """The pairs this token unlocks, or ``None`` if it matches no configured token.

    Compared with ``compare_digest`` against every entry rather than looked up, so
    a token is checked the same way the internal key is. ``None`` is distinct from
    an empty set, which is a configured token that currently unlocks nothing.
    """
    for candidate, allowed in table.items():
        if hmac.compare_digest(token.encode("utf-8"), candidate.encode("utf-8")):
            return allowed
    return None


def is_internal(
    x_internal_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> bool:
    """True only if an internal key is configured and the presented key matches it."""
    expected = settings.internal_api_key
    if expected is None or x_internal_key is None:
        return False
    return hmac.compare_digest(
        x_internal_key.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    )


def _fingerprint(token: str) -> str:
    """A short, non-reversible tag for correlating a rejected token in logs."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def hidden_early_access(
    response: Response,
    internal: bool = Depends(is_internal),
    x_ea_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> frozenset[tuple[str, str]]:
    """The pairs this caller's responses must not contain.

    Cache headers are set here rather than per route, so no endpoint can serve
    embargoed rows without them. Several proofs may be presented at once; the
    caller gets the union of what they unlock.
    """
    # Appended, not assigned: SelectiveGZipMiddleware adds Accept-Encoding after
    # the route returns, and assignment here would be overwritten.
    response.headers.append("Vary", VARY_HEADERS)

    if internal:
        response.headers[EA_STATUS_HEADER] = "internal"
        response.headers["Cache-Control"] = "private, no-store"
        return frozenset()

    embargoed = embargoed_pairs()

    # Exclusive orgs are resolved first and never widened: an X-EA-Token must not
    # add to a view whose whole point is what it leaves out.
    if authorization is not None and settings.clerk_issuer is not None:
        universe = all_registered_pairs()
        only = clerk.exclusive_pairs(authorization, settings, universe)
        if only is not None:
            response.headers[EA_STATUS_HEADER] = "accepted"
            response.headers["Cache-Control"] = "private, no-store"
            return universe - with_retired_keys(only)

    unlocked: frozenset[tuple[str, str]] | None = None
    if x_ea_token is not None:
        allowed = _allowed_for(x_ea_token, _allowlists(settings))
        if allowed is None:
            logger.warning("early_access_token_unknown", token=_fingerprint(x_ea_token))
        else:
            unlocked = with_retired_keys(allowed)

    if authorization is not None and settings.clerk_issuer is not None:
        bearer = clerk.allowed_pairs(authorization, settings, embargoed)
        if bearer is not None:
            unlocked = (unlocked or frozenset()) | with_retired_keys(bearer)
    elif x_ea_token is None:
        response.headers[EA_STATUS_HEADER] = "absent"
        return embargoed

    if unlocked is None:
        response.headers[EA_STATUS_HEADER] = "unknown"
        return embargoed

    response.headers[EA_STATUS_HEADER] = "accepted"
    response.headers["Cache-Control"] = "private, no-store"
    return embargoed - unlocked
