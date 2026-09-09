# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify Google-signed identity tokens; proves who is calling, not that they are an admin."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import jwt
import structlog

from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.api.google_auth")

JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})


@dataclass(frozen=True)
class GoogleIdentity:
    sub: str
    email: str


@functools.lru_cache(maxsize=1)
def _jwks() -> jwt.PyJWKClient:
    return jwt.PyJWKClient(JWKS_URL)


def looks_like_google(token: str) -> bool:
    try:
        claims = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return False
    return claims.get("iss") in ISSUERS


def verify(token: str, settings: Settings) -> GoogleIdentity | None:
    if not settings.admin_google_audiences:
        logger.warning("google_token_rejected", error="admin_google_audiences unset")
        return None
    try:
        key = _jwks().get_signing_key_from_jwt(token).key
        claims: dict[str, Any] = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=settings.admin_google_audiences,
            issuer=ISSUERS,
            leeway=5,
            options={"require": ["exp", "sub", "email"]},
        )
    except jwt.PyJWTError as exc:
        logger.warning("google_token_rejected", error=str(exc))
        return None
    sub = claims.get("sub")
    email = claims.get("email")
    if not isinstance(sub, str) or not sub or not isinstance(email, str) or not email:
        logger.warning("google_token_rejected", error="sub or email malformed")
        return None
    if claims.get("email_verified") is not True:
        logger.warning("google_token_rejected", error="email not verified")
        return None
    return GoogleIdentity(sub=sub, email=email.lower())
