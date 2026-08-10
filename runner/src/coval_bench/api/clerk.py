# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a Clerk session token to the embargoed models its holder may see.

Verified against the instance JWKS (public — no secret involved). A mapped
``org_id`` scopes the caller to its provider — even for coval.dev emails, so
switching into a partner org previews exactly their view. Outside a mapped
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


@functools.lru_cache(maxsize=1)
def _parse_org_providers(raw: str) -> Mapping[str, str]:
    """Parse the ``{org_id: provider}`` settings blob; a malformed blob maps nothing."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("clerk_org_providers_unparsable")
        return {}
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
    ):
        logger.error("clerk_org_providers_malformed")
        return {}
    return parsed


def _org_provider(claims: dict[str, Any], settings: Settings) -> str | None:
    org_id = claims.get("org_id")
    if not isinstance(org_id, str) or settings.clerk_org_providers is None:
        return None
    return _parse_org_providers(settings.clerk_org_providers).get(org_id)


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
    provider = _org_provider(claims, settings)
    if provider is not None:
        return frozenset(pair for pair in embargoed if pair[0] == provider)
    email = claims.get("email")
    if isinstance(email, str) and email.endswith(_INTERNAL_EMAIL_SUFFIX):
        return embargoed
    return frozenset()
