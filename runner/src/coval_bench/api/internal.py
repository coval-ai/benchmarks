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

# Both proof headers. Listing one is worse than listing none: a cached internal
# response carries no X-EA-Token, so it would match a public request.
VARY_HEADERS = "X-Internal-Key, X-EA-Token"


def never_shared(response: Response) -> None:
    """Mark a response as belonging to this caller alone.

    Any response whose content depends on the presented proof needs this, errors
    included: the same URL is a 404 for the public and a redirect for a partner,
    so a shared cache must never hand one caller's answer to another.
    """
    response.headers.append("Vary", "X-Internal-Key, X-EA-Token")
    response.headers["Cache-Control"] = "private, no-store"


def embargoed_pairs() -> frozenset[tuple[str, str]]:
    """Every ``(provider, model)`` pair currently under embargo."""
    return frozenset(
        (m.provider, m.model) for m in MODEL_REGISTRY if m.status is ModelStatus.EARLY_ACCESS
    )


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

    Sets the cache headers here rather than per route, so no endpoint can serve
    embargoed rows without them. Only a presented-but-unknown token is logged; an
    absent one is ordinary public traffic. Subtracting the allowlist means an entry
    naming a model that is no longer embargoed is inert, and no token reveals what
    the registry does not currently embargo.
    """
    # Appended, not assigned: SelectiveGZipMiddleware adds Accept-Encoding after
    # the route returns, and assignment here would be overwritten.
    response.headers.append("Vary", VARY_HEADERS)

    if internal:
        response.headers[EA_STATUS_HEADER] = "internal"
        response.headers["Cache-Control"] = "private, no-store"
        return frozenset()

    embargoed = embargoed_pairs()
    if x_ea_token is None and authorization is not None and settings.clerk_issuer is not None:
        allowed = clerk.allowed_pairs(authorization, settings, embargoed)
        if allowed is None:
            response.headers[EA_STATUS_HEADER] = "unknown"
            return embargoed
        response.headers[EA_STATUS_HEADER] = "accepted"
        return embargoed - allowed

    if x_ea_token is None:
        response.headers[EA_STATUS_HEADER] = "absent"
        return embargoed

    allowed = _allowed_for(x_ea_token, _allowlists(settings))
    if allowed is None:
        # The one case worth alerting on: a link that should work and doesn't.
        logger.warning("early_access_token_unknown", token=_fingerprint(x_ea_token))
        response.headers[EA_STATUS_HEADER] = "unknown"
        return embargoed

    response.headers[EA_STATUS_HEADER] = "accepted"
    response.headers["Cache-Control"] = "private, no-store"
    return embargoed - allowed
