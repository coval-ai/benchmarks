# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The admin auth dependency: 401 without a proven token, 403 outside the coval org."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from coval_bench.api import clerk
from coval_bench.api.deps import require_coval_admin
from coval_bench.config import Settings
from tests.api.conftest import (
    _CLERK_PUBLIC_PEM,
    CLERK_ISSUER,
    CLERK_PARTY,
    COVAL_ORG,
    mint_clerk_token,
)


@pytest.fixture(autouse=True)
def _stub_jwks(monkeypatch: pytest.MonkeyPatch) -> None:
    signing_key = SimpleNamespace(key=_CLERK_PUBLIC_PEM)
    client = SimpleNamespace(get_signing_key_from_jwt=lambda token: signing_key)
    monkeypatch.setattr(clerk, "_jwks", lambda issuer: client)


def _settings(monkeypatch: pytest.MonkeyPatch, coval_org: str | None = COVAL_ORG) -> Settings:
    monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "stt-v1")
    monkeypatch.setenv("CLERK_ISSUER", CLERK_ISSUER)
    monkeypatch.setenv("CLERK_AUTHORIZED_PARTIES", json.dumps([CLERK_PARTY]))
    if coval_org is None:
        monkeypatch.delenv("CLERK_COVAL_ORG", raising=False)
    else:
        monkeypatch.setenv("CLERK_COVAL_ORG", coval_org)
    return Settings()


def _admin(settings: Settings, authorization: str | None) -> clerk.CovalAdmin:
    return require_coval_admin(authorization=authorization, settings=settings)


def _bearer(**claims: Any) -> str:
    return f"Bearer {mint_clerk_token(**claims)}"


@pytest.mark.parametrize("authorization", [None, "", "Token abc", "Bearer ", "Bearer not-a-jwt"])
def test_unproven_tokens_are_401(
    monkeypatch: pytest.MonkeyPatch, authorization: str | None
) -> None:
    settings = _settings(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        _admin(settings, authorization)
    assert exc.value.status_code == 401
    assert exc.value.headers is not None
    assert exc.value.headers["WWW-Authenticate"] == "Bearer"


def test_expired_token_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    now = int(time.time())
    header = _bearer(sub="user_1", org_id=COVAL_ORG, iat=now - 120, exp=now - 60)
    with pytest.raises(HTTPException) as exc:
        _admin(settings, header)
    assert exc.value.status_code == 401


def test_token_without_a_subject_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        _admin(settings, _bearer(org_id=COVAL_ORG))
    assert exc.value.status_code == 401


@pytest.mark.parametrize(
    "claims",
    [
        {},
        {"org_id": "org_test_first"},
        {"org_id": COVAL_ORG + "x"},
        {"org_id": [COVAL_ORG]},
    ],
)
def test_tokens_outside_the_coval_org_are_403(
    monkeypatch: pytest.MonkeyPatch, claims: dict[str, Any]
) -> None:
    settings = _settings(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        _admin(settings, _bearer(sub="user_1", **claims))
    assert exc.value.status_code == 403


def test_unset_coval_org_is_403_for_everyone(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, coval_org=None)
    with pytest.raises(HTTPException) as exc:
        _admin(settings, _bearer(sub="user_1", org_id=COVAL_ORG))
    assert exc.value.status_code == 403


def test_coval_org_token_yields_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    admin = _admin(settings, _bearer(sub="user_1", org_id=COVAL_ORG, email="cale@coval.dev"))
    assert admin == clerk.CovalAdmin(user_id="user_1", org_id=COVAL_ORG, email="cale@coval.dev")


@pytest.mark.parametrize("claims", [{}, {"email": ""}, {"email": 7}])
def test_a_missing_blank_or_malformed_email_is_none(
    monkeypatch: pytest.MonkeyPatch, claims: dict[str, Any]
) -> None:
    settings = _settings(monkeypatch)
    admin = _admin(settings, _bearer(sub="user_1", org_id=COVAL_ORG, **claims))
    assert admin.email is None
