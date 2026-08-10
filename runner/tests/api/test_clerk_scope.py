# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider-org early access via Clerk session tokens (JWKS stubbed).

Assertions derive the embargoed set from the registry, so they keep holding
as the roster changes status.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Response

from coval_bench.api import clerk
from coval_bench.api.internal import (
    EA_STATUS_HEADER,
    embargoed_pairs,
    hidden_early_access,
    with_retired_keys,
)
from coval_bench.config import Settings

_ISSUER = "https://clerk.example.com"
_PARTY = "https://benchmarks.example.com"

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_PEM = _PRIVATE_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)

_INTERNAL_EMAIL = "someone@coval.dev"
_ORG_ID = "org_2abc123"


@pytest.fixture(autouse=True)
def _stub_jwks(monkeypatch: pytest.MonkeyPatch) -> None:
    signing_key = SimpleNamespace(key=_PUBLIC_PEM)
    client = SimpleNamespace(get_signing_key_from_jwt=lambda token: signing_key)
    monkeypatch.setattr(clerk, "_jwks", lambda issuer: client)


def _settings(
    monkeypatch: pytest.MonkeyPatch,
    issuer: str | None = _ISSUER,
    parties: str | None = f'["{_PARTY}"]',
    ea_tokens: str | None = None,
    org_providers: str | None = None,
) -> Settings:
    monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "stt-v1")
    monkeypatch.setenv("RUNNER_SHA", "test-sha")
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    if ea_tokens is None:
        monkeypatch.delenv("EARLY_ACCESS_TOKENS", raising=False)
    else:
        monkeypatch.setenv("EARLY_ACCESS_TOKENS", ea_tokens)
    if org_providers is None:
        monkeypatch.delenv("CLERK_ORG_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("CLERK_ORG_PROVIDERS", org_providers)
    if issuer is None:
        monkeypatch.delenv("CLERK_ISSUER", raising=False)
    else:
        monkeypatch.setenv("CLERK_ISSUER", issuer)
    if parties is None:
        monkeypatch.delenv("CLERK_AUTHORIZED_PARTIES", raising=False)
    else:
        monkeypatch.setenv("CLERK_AUTHORIZED_PARTIES", parties)
    return Settings()


def _mint(claims: dict[str, Any], key: rsa.RSAPrivateKey = _PRIVATE_KEY, **overrides: Any) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {"iss": _ISSUER, "iat": now, "exp": now + 60, "azp": _PARTY}
    payload.update(claims)
    payload.update(overrides)
    return jwt.encode(payload, key, algorithm="RS256")


def _resolve(
    settings: Settings, authorization: str | None, x_ea_token: str | None = None
) -> tuple[frozenset[tuple[str, str]], str]:
    """This caller's hidden set and the status header it got."""
    response = Response()
    hidden = hidden_early_access(
        response=response,
        internal=False,
        x_ea_token=x_ea_token,
        authorization=authorization,
        settings=settings,
    )
    return hidden, response.headers[EA_STATUS_HEADER]


def _embargoed_provider() -> str:
    """Any provider with an embargoed model, or skip."""
    providers = sorted({provider for provider, _ in embargoed_pairs()})
    if not providers:
        pytest.skip("no embargoed models in the registry")
    return providers[0]


def _two_embargoed_providers() -> tuple[str, str]:
    """Two distinct providers with embargoed models, or skip."""
    providers = sorted({provider for provider, _ in embargoed_pairs()})
    if len(providers) < 2:
        pytest.skip("fewer than two providers have embargoed models")
    return providers[0], providers[1]


def test_org_sees_its_own_providers_models_and_no_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mine, theirs = _two_embargoed_providers()
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: mine}))
    token = _mint({"org_id": _ORG_ID})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == frozenset(pair for pair in embargoed_pairs() if pair[0] != mine)
    assert all(provider != mine for provider, _ in hidden)
    assert any(provider == theirs for provider, _ in hidden)


def test_coval_email_sees_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"email": _INTERNAL_EMAIL})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == frozenset()


def test_mapped_org_scopes_even_a_coval_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching into a partner org previews exactly that org's view."""
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: provider}))
    token = _mint({"email": _INTERNAL_EMAIL, "org_id": _ORG_ID})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == frozenset(pair for pair in embargoed_pairs() if pair[0] != provider)


def test_coval_email_in_an_unmapped_org_still_sees_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a mapped org narrows the view; an unmapped one must not demote to public."""
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: provider}))
    token = _mint({"email": _INTERNAL_EMAIL, "org_id": "org_2someoneelse"})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == frozenset()


def test_lookalike_email_domain_unlocks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the exact @coval.dev suffix is internal, not a domain ending in it."""
    settings = _settings(monkeypatch)
    token = _mint({"email": "someone@notcoval.dev"})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_session_without_the_claims_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_signed_in_user_without_an_org_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"email": "user@example.com", "org_id": None})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_unmapped_org_keeps_the_public_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real org unlocks nothing until settings say which provider it is."""
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: provider}))
    token = _mint({"org_id": "org_2someoneelse"})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_org_without_a_configured_map_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"org_id": _ORG_ID})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_claims_of_the_wrong_shape_unlock_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: provider}))
    token = _mint({"email": [_INTERNAL_EMAIL], "org_id": [_ORG_ID]})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_expired_token_keeps_the_public_view(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    now = int(time.time())
    token = _mint({"email": _INTERNAL_EMAIL}, iat=now - 120, exp=now - 60)
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_token_signed_by_another_key_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"email": _INTERNAL_EMAIL}, key=_OTHER_KEY)
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_token_from_another_issuer_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"email": _INTERNAL_EMAIL}, iss="https://not-clerk.example.com")
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_azp_outside_the_allowlist_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, parties=f'["{_PARTY}"]')
    token = _mint({"email": _INTERNAL_EMAIL}, azp="https://elsewhere.example.com")
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_azp_in_the_allowlist_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, parties=f'["{_PARTY}"]')
    token = _mint({"email": _INTERNAL_EMAIL})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == frozenset()


def test_unset_authorized_parties_rejects_every_token(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, parties=None)
    token = _mint({"email": _INTERNAL_EMAIL})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def _headers(settings: Settings, authorization: str) -> dict[str, str]:
    response = Response()
    hidden_early_access(
        response=response,
        internal=False,
        x_ea_token=None,
        authorization=authorization,
        settings=settings,
    )
    return dict(response.headers)


def test_accepted_bearer_response_is_never_stored_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"email": _INTERNAL_EMAIL})
    headers = _headers(settings, f"Bearer {token}")
    assert headers["cache-control"] == "private, no-store"


def test_bearer_responses_vary_on_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"email": _INTERNAL_EMAIL})
    headers = _headers(settings, f"Bearer {token}")
    assert "Authorization" in headers["vary"]


def test_bearer_is_ignored_when_clerk_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, issuer=None)
    token = _mint({"email": _INTERNAL_EMAIL})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "absent"
    assert hidden == embargoed_pairs()


def test_non_bearer_authorization_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    hidden, status = _resolve(settings, "Basic dXNlcjpwYXNz")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_partner_token_and_bearer_unlock_the_union(monkeypatch: pytest.MonkeyPatch) -> None:
    """One request may carry both proofs, say an embargo link opened while signed in."""
    mine, theirs = _two_embargoed_providers()
    their_pair = min(pair for pair in embargoed_pairs() if pair[0] == theirs)
    settings = _settings(
        monkeypatch,
        ea_tokens=json.dumps({"partner-token": [f"{their_pair[0]}/{their_pair[1]}"]}),
        org_providers=json.dumps({_ORG_ID: mine}),
    )
    token = _mint({"org_id": _ORG_ID})
    hidden, status = _resolve(settings, f"Bearer {token}", x_ea_token="partner-token")  # noqa: S106 - fake token
    assert status == "accepted"
    unlocked = with_retired_keys(
        frozenset({their_pair}) | frozenset(pair for pair in embargoed_pairs() if pair[0] == mine)
    )
    assert hidden == embargoed_pairs() - unlocked


def test_valid_bearer_rescues_an_unknown_partner_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"email": _INTERNAL_EMAIL})
    hidden, status = _resolve(settings, f"Bearer {token}", x_ea_token="not-a-configured-token")  # noqa: S106 - fake token
    assert status == "accepted"
    assert hidden == frozenset()


def test_valid_partner_token_survives_a_bad_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _embargoed_provider()
    pair = min(p for p in embargoed_pairs() if p[0] == provider)
    settings = _settings(
        monkeypatch,
        ea_tokens=json.dumps({"partner-token": [f"{pair[0]}/{pair[1]}"]}),
    )
    now = int(time.time())
    expired = _mint({"org_id": _ORG_ID}, iat=now - 120, exp=now - 60)
    hidden, status = _resolve(settings, f"Bearer {expired}", x_ea_token="partner-token")  # noqa: S106 - fake token
    assert status == "accepted"
    assert hidden == embargoed_pairs() - with_retired_keys(frozenset({pair}))


def test_two_bad_proofs_stay_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    now = int(time.time())
    expired = _mint({"email": _INTERNAL_EMAIL}, iat=now - 120, exp=now - 60)
    hidden, status = _resolve(settings, f"Bearer {expired}", x_ea_token="not-a-configured-token")  # noqa: S106 - fake token
    assert status == "unknown"
    assert hidden == embargoed_pairs()
