# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a Clerk session token to the embargoed models its org may see.

Verified against the instance JWKS (public — no secret involved). The
``benchmark_providers`` claim names the org's providers, or ``"*"`` for all.
"""

from __future__ import annotations

import functools
from typing import Any

import jwt
import structlog

from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.api.clerk")


@functools.lru_cache(maxsize=4)
def _jwks(issuer: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"{issuer}/.well-known/jwks.json")


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
    providers = claims.get("benchmark_providers")
    if providers == "*":
        return embargoed
    if not isinstance(providers, list):
        return frozenset()
    return frozenset(pair for pair in embargoed if pair[0] in providers)
