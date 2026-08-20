# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a Clerk session token to the embargoed models its holder may see.

Verified against the instance JWKS (public — no secret involved). A mapped
``org_id`` scopes the caller to what its entry names — even for coval.dev emails,
so switching into a partner org previews exactly their view. Outside a mapped
org, a coval.dev ``email`` sees everything.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from typing import Any

import jwt
import structlog

from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.api.clerk")

_INTERNAL_EMAIL_SUFFIX = "@coval.dev"


@functools.lru_cache(maxsize=4)
def _jwks(issuer: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"{issuer}/.well-known/jwks.json")


@functools.lru_cache(maxsize=2)
def _parse_org_entries(raw: str) -> Mapping[str, tuple[str, ...]]:
    """Parse the ``{org_id: "provider" | ["provider/model", ...]}`` settings blob.

    A malformed entry is dropped rather than guessed at, so a bad deploy narrows a
    view instead of widening one.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("clerk_org_providers_unparsable")
        return {}
    if not isinstance(parsed, dict):
        logger.error("clerk_org_providers_malformed")
        return {}
    table: dict[str, tuple[str, ...]] = {}
    for org_id, value in parsed.items():
        entries = [value] if isinstance(value, str) else value
        if (
            not isinstance(org_id, str)
            or not isinstance(entries, list)
            or not all(isinstance(entry, str) and entry for entry in entries)
        ):
            logger.error("clerk_org_providers_malformed")
            if isinstance(org_id, str):
                table[org_id] = ()
            continue
        table[org_id] = tuple(entries)
    return table


@functools.lru_cache(maxsize=2)
def _parse_exclusive(raw: str) -> Mapping[str, tuple[str, ...]] | None:
    """As ``_parse_org_entries``, but ``None`` when the blob itself is unusable."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("clerk_org_exclusive_unparsable")
        return None
    if not isinstance(parsed, dict):
        logger.error("clerk_org_exclusive_malformed")
        return None
    return _parse_org_entries(raw)


def _org_unlocked(
    claims: dict[str, Any], settings: Settings, embargoed: frozenset[tuple[str, str]]
) -> frozenset[tuple[str, str]] | None:
    """The pairs this token's org unlocks, or ``None`` when its org is unmapped.

    An entry is a whole provider, or one ``provider/model`` pair. A mapped org that
    names nothing unlocks nothing, which is distinct from being unmapped.
    """
    org_id = claims.get("org_id")
    if not isinstance(org_id, str) or settings.clerk_org_providers is None:
        return None
    entries = _parse_org_entries(settings.clerk_org_providers).get(org_id)
    if entries is None:
        return None
    return _expand(entries, embargoed, org_id=org_id, scope="providers")


def _expand(
    entries: tuple[str, ...],
    universe: frozenset[tuple[str, str]],
    *,
    org_id: str,
    scope: str,
) -> frozenset[tuple[str, str]]:
    """The pairs of *universe* that *entries* name, by provider or provider/model.

    An entry naming nothing is a typo or a retired model, and silently grants zero,
    so it is logged with the org it came from. Never log the token itself.
    """
    named: set[tuple[str, str]] = set()
    for entry in entries:
        provider, sep, model = entry.partition("/")
        matched = {
            pair for pair in universe if (pair == (provider, model) if sep else pair[0] == provider)
        }
        if not matched:
            logger.warning("clerk_org_entry_unmatched", org_id=org_id, entry=entry, scope=scope)
        named |= matched
    return frozenset(named)


def exclusive_pairs(
    authorization: str, settings: Settings, universe: frozenset[tuple[str, str]]
) -> frozenset[tuple[str, str]] | None:
    """The only pairs this token's org may see, or ``None`` when it is not exclusive.

    An exclusive org sees nothing else at all, public models included, so this is
    resolved against every known pair rather than against the embargoed ones.
    """
    if settings.clerk_org_exclusive is None:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    claims = _claims(token.strip(), settings)
    if claims is None:
        return None
    org_id = claims.get("org_id")
    if not isinstance(org_id, str):
        return None
    table = _parse_exclusive(settings.clerk_org_exclusive)
    if table is None:
        # Configured but unusable. Denying every org-bearing token is deliberate: the
        # blob is what says who is restricted, so without it we cannot tell, and
        # serving ordinary visibility would be the one unrecoverable answer. Startup
        # validation in Settings should make this unreachable.
        logger.error("clerk_org_exclusive_denied_all")
        return frozenset()
    entries = table.get(org_id)
    if entries is None:
        return None
    return _expand(entries, universe, org_id=org_id, scope="exclusive")


def _claims(token: str, settings: Settings) -> dict[str, Any] | None:
    """Verified claims, or ``None`` if the token proves nothing."""
    issuer = settings.clerk_issuer
    parties = settings.clerk_authorized_parties
    if issuer is None or not parties:
        logger.warning("clerk_token_rejected", error="clerk_authorized_parties unset")
        return None
    try:
        key = _jwks(issuer).get_signing_key_from_jwt(token).key
        claims: dict[str, Any] = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=issuer,
            leeway=5,
            options={"require": ["exp"]},
        )
    except jwt.PyJWTError as exc:
        logger.warning("clerk_token_rejected", error=str(exc))
        return None
    if claims.get("azp") not in parties:
        logger.warning("clerk_token_rejected", error="azp not in clerk_authorized_parties")
        return None
    return claims


def internal_email(authorization: str | None, settings: Settings) -> str | None:
    """The verified coval.dev email this token proves, or ``None``.

    The admin gate: unlike the data endpoints' visibility rules, an active
    partner org never narrows it — a coval dev previewing a partner view can
    still administer. Anything short of a verified internal email is ``None``.
    """
    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    claims = _claims(token.strip(), settings)
    if claims is None:
        return None
    email = claims.get("email")
    if isinstance(email, str) and email.endswith(_INTERNAL_EMAIL_SUFFIX):
        return email
    return None


def allowed_pairs(
    authorization: str, settings: Settings, embargoed: frozenset[tuple[str, str]]
) -> frozenset[tuple[str, str]] | None:
    """The embargoed pairs this bearer token unlocks, or ``None`` if it proves nothing."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    claims = _claims(token.strip(), settings)
    if claims is None:
        return None
    org_unlocked = _org_unlocked(claims, settings, embargoed)
    if org_unlocked is not None:
        return org_unlocked
    email = claims.get("email")
    if isinstance(email, str) and email.endswith(_INTERNAL_EMAIL_SUFFIX):
        return embargoed
    return frozenset()
