# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticated opaque cursors for the public normalized-results reader."""

from __future__ import annotations

import json

from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

CURSOR_PREFIX = "v1."
MAX_CURSOR_LENGTH = 8192


class CursorError(ValueError):
    """The token is malformed, unauthenticated, or too large."""


class CursorKeyError(ValueError):
    """The server cursor key is absent or not a valid Fernet key."""


def _fernet(key: SecretStr | None) -> Fernet:
    if key is None:
        raise CursorKeyError("results cursor key is not configured")
    raw = key.get_secret_value()
    if not raw:
        raise CursorKeyError("results cursor key is empty")
    try:
        return Fernet(raw.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise CursorKeyError("results cursor key is invalid") from exc


def encode(payload: dict[str, object], key: SecretStr | None) -> str:
    token = CURSOR_PREFIX + _fernet(key).encrypt(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    if len(token) > MAX_CURSOR_LENGTH:
        raise CursorError("cursor is too large")
    return token


def validate_key(key: SecretStr | None) -> SecretStr:
    """Validate a configured key before other route dependencies can run."""
    _fernet(key)
    if key is None:  # pragma: no cover - _fernet raises first
        raise CursorKeyError("results cursor key is not configured")
    return key


def decode(token: str, key: SecretStr | None) -> dict[str, object]:
    if len(token) > MAX_CURSOR_LENGTH or not token.startswith(CURSOR_PREFIX):
        raise CursorError("invalid cursor")
    try:
        raw = _fernet(key).decrypt(token[len(CURSOR_PREFIX) :].encode("ascii"))
        payload = json.loads(raw)
    except CursorKeyError:
        raise
    except (InvalidToken, ValueError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CursorError("invalid cursor") from exc
    if not isinstance(payload, dict):
        raise CursorError("invalid cursor")
    return payload
