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
from pydantic import ValidationError
from structlog.testing import capture_logs

from coval_bench.api import clerk
from coval_bench.api.internal import (
    EA_STATUS_HEADER,
    all_registered_pairs,
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

_COVAL_ORG = "org_2coval"
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
    org_providers: str | None = None,
    org_exclusive: str | None = None,
    coval_org: str | None = _COVAL_ORG,
) -> Settings:
    monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "stt-v1")
    if org_providers is None:
        monkeypatch.delenv("CLERK_ORG_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("CLERK_ORG_PROVIDERS", org_providers)
    if org_exclusive is None:
        monkeypatch.delenv("CLERK_ORG_EXCLUSIVE", raising=False)
    else:
        monkeypatch.setenv("CLERK_ORG_EXCLUSIVE", org_exclusive)
    if coval_org is None:
        monkeypatch.delenv("CLERK_COVAL_ORG", raising=False)
    else:
        monkeypatch.setenv("CLERK_COVAL_ORG", coval_org)
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
    settings: Settings, authorization: str | None
) -> tuple[frozenset[tuple[str, str]], str]:
    """This caller's hidden set and the status header it got."""
    response = Response()
    hidden = hidden_early_access(
        response=response,
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


def test_org_entry_can_name_one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = sorted(embargoed_pairs())[0]
    provider, model = pair
    siblings = {p for p in embargoed_pairs() if p[0] == provider and p != pair}
    if not siblings:
        pytest.skip("no embargoed provider has two models")
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: [f"{provider}/{model}"]}))
    token = _mint({"org_id": _ORG_ID})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert pair not in hidden
    assert siblings <= hidden


def test_org_entry_list_of_one_provider_unlocks_all_its_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: [provider]}))
    token = _mint({"org_id": _ORG_ID})
    hidden, _ = _resolve(settings, f"Bearer {token}")
    assert all(p != provider for p, _ in hidden)


def test_mapped_org_naming_nothing_unlocks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: []}))
    token = _mint({"org_id": _ORG_ID})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == with_retired_keys(embargoed_pairs())


def test_malformed_org_entry_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: [provider, 7]}))
    token = _mint({"org_id": _ORG_ID, "email": "partner@example.com"})
    hidden, _ = _resolve(settings, f"Bearer {token}")
    assert hidden == with_retired_keys(embargoed_pairs())


def _public_pair() -> tuple[str, str]:
    """Any registered pair that is not embargoed, or skip."""
    public = sorted(all_registered_pairs() - embargoed_pairs())
    if not public:
        pytest.skip("every registered model is embargoed")
    return public[0]


def test_exclusive_org_sees_only_its_own_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _embargoed_provider()
    mine = {pair for pair in embargoed_pairs() if pair[0] == provider}
    settings = _settings(monkeypatch, org_exclusive=json.dumps({_ORG_ID: [provider]}))
    token = _mint({"org_id": _ORG_ID})
    hidden, status = _resolve(settings, f"Bearer {token}")

    assert status == "accepted"
    assert not (mine & hidden)
    assert _public_pair() in hidden
    assert hidden == all_registered_pairs() - with_retired_keys(frozenset(mine))


def test_exclusive_org_can_name_one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, model = sorted(embargoed_pairs())[0]
    settings = _settings(monkeypatch, org_exclusive=json.dumps({_ORG_ID: [f"{provider}/{model}"]}))
    token = _mint({"org_id": _ORG_ID})
    hidden, _ = _resolve(settings, f"Bearer {token}")

    assert (provider, model) not in hidden
    assert len(all_registered_pairs() - hidden) == 1


def test_exclusive_wins_over_additive_for_the_same_org(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _embargoed_provider()
    settings = _settings(
        monkeypatch,
        org_providers=json.dumps({_ORG_ID: [provider]}),
        org_exclusive=json.dumps({_ORG_ID: [provider]}),
    )
    token = _mint({"org_id": _ORG_ID})
    hidden, _ = _resolve(settings, f"Bearer {token}")

    assert _public_pair() in hidden


def test_exclusive_overrides_the_coval_org(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_exclusive=json.dumps({_COVAL_ORG: [provider]}))
    token = _mint({"org_id": _COVAL_ORG})
    hidden, _ = _resolve(settings, f"Bearer {token}")

    assert _public_pair() in hidden


def test_unmapped_org_is_unaffected_by_the_exclusive_map(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_exclusive=json.dumps({"org_someone_else": [provider]}))
    token = _mint({"org_id": _COVAL_ORG})
    hidden, _ = _resolve(settings, f"Bearer {token}")

    assert hidden == frozenset()


@pytest.mark.parametrize(
    "blob",
    [
        '{"org_x": ["colors"]',
        '{"org_x": 7}',
        '{"org_x": [""]}',
        '{"org_x": ""}',
        '{"": ["colors"]}',
        '["colors"]',
    ],
)
def test_unusable_exclusive_map_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, blob: str
) -> None:
    with pytest.raises(ValidationError):
        _settings(monkeypatch, org_exclusive=blob)


def _unvalidated_settings(exclusive: str) -> Settings:
    """Settings carrying a blob the field validator would have rejected."""
    return Settings.model_construct(
        clerk_issuer=_ISSUER,
        clerk_authorized_parties=[_PARTY],
        clerk_org_providers=None,
        clerk_org_exclusive=exclusive,
    )


def test_unparsable_exclusive_map_denies_instead_of_falling_back() -> None:
    settings = _unvalidated_settings('{"org_x": ["colors"]')
    token = _mint({"org_id": _ORG_ID})
    hidden, status = _resolve(settings, f"Bearer {token}")

    assert status == "accepted"
    assert hidden == all_registered_pairs()
    assert _public_pair() in hidden


def test_malformed_org_entry_in_exclusive_map_denies() -> None:
    settings = _unvalidated_settings(json.dumps({_ORG_ID: [7]}))
    token = _mint({"org_id": _ORG_ID})
    hidden, _ = _resolve(settings, f"Bearer {token}")

    assert hidden == all_registered_pairs()


def test_unknown_exclusive_entry_grants_nothing_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, org_exclusive=json.dumps({_ORG_ID: ["colours/gray"]}))
    token = _mint({"org_id": _ORG_ID})
    response = Response()
    with capture_logs() as logs:
        hidden = hidden_early_access(
            response=response,
            authorization=f"Bearer {token}",
            settings=settings,
        )

    assert hidden == all_registered_pairs()
    warned = [entry for entry in logs if entry["event"] == "clerk_org_entry_unmatched"]
    assert [(entry["org_id"], entry["entry"], entry["scope"]) for entry in warned] == [
        (_ORG_ID, "colours/gray", "exclusive")
    ]
    assert not any("Bearer" in str(entry) for entry in logs)


def test_coval_org_sees_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"org_id": _COVAL_ORG})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == frozenset()


def test_unset_coval_org_grants_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, coval_org=None)
    token = _mint({"org_id": _COVAL_ORG})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_a_mapped_coval_org_gets_only_its_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mapping the coval org in clerk_org_providers narrows it like any other org."""
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_providers=json.dumps({_COVAL_ORG: provider}))
    token = _mint({"org_id": _COVAL_ORG})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == frozenset(pair for pair in embargoed_pairs() if pair[0] != provider)


def test_a_coval_email_alone_unlocks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Staff identity is the coval org, not the mailbox domain."""
    settings = _settings(monkeypatch)
    token = _mint({"email": "someone@coval.dev"})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_a_coval_email_in_another_org_unlocks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staff view requires the coval org active, whatever the email says."""
    provider = _embargoed_provider()
    settings = _settings(monkeypatch, org_providers=json.dumps({_ORG_ID: provider}))
    token = _mint({"email": "someone@coval.dev", "org_id": "org_2someoneelse"})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_a_lookalike_org_id_unlocks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the exact coval org id is staff, not a superstring of it."""
    settings = _settings(monkeypatch)
    token = _mint({"org_id": f"{_COVAL_ORG}x"})
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
    token = _mint({"org_id": [_COVAL_ORG]})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == embargoed_pairs()


def test_expired_token_keeps_the_public_view(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    now = int(time.time())
    token = _mint({"org_id": _COVAL_ORG}, iat=now - 120, exp=now - 60)
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_token_signed_by_another_key_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"org_id": _COVAL_ORG}, key=_OTHER_KEY)
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_token_from_another_issuer_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"org_id": _COVAL_ORG}, iss="https://not-clerk.example.com")
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_azp_outside_the_allowlist_keeps_the_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, parties=f'["{_PARTY}"]')
    token = _mint({"org_id": _COVAL_ORG}, azp="https://elsewhere.example.com")
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def test_azp_in_the_allowlist_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, parties=f'["{_PARTY}"]')
    token = _mint({"org_id": _COVAL_ORG})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "accepted"
    assert hidden == frozenset()


def test_unset_authorized_parties_rejects_every_token(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, parties=None)
    token = _mint({"org_id": _COVAL_ORG})
    hidden, status = _resolve(settings, f"Bearer {token}")
    assert status == "unknown"
    assert hidden == embargoed_pairs()


def _headers(settings: Settings, authorization: str) -> dict[str, str]:
    response = Response()
    hidden_early_access(response=response, authorization=authorization, settings=settings)
    return dict(response.headers)


def test_accepted_bearer_response_is_never_stored_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"org_id": _COVAL_ORG})
    headers = _headers(settings, f"Bearer {token}")
    assert headers["cache-control"] == "private, no-store"


def test_bearer_responses_vary_on_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    token = _mint({"org_id": _COVAL_ORG})
    headers = _headers(settings, f"Bearer {token}")
    assert "Authorization" in headers["vary"]


def test_bearer_is_ignored_when_clerk_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, issuer=None)
    token = _mint({"org_id": _COVAL_ORG})
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


def test_retired_board_keys_stay_embargoed() -> None:
    """A renamed embargoed model keeps hiding artefacts stored under its old key.

    Sample manifests and result rows written before the rename still carry the
    old string, and the embargo matches on what is stored -- so losing the old
    key would publish every recording and row published under that name.
    """
    assert ("xai", "grok-realtime") in embargoed_pairs()
    assert ("xai", "grok-voice-think-fast-1.0") in embargoed_pairs()


def test_org_grant_for_a_renamed_model_also_sees_its_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing an org for a model clears it for its pre-rename artefacts.

    A grant names models, not the strings they were once published under, so an
    org cleared for the current key must still reach the manifests that name the
    old one -- otherwise a rename silently narrows what that org can see.
    """
    settings = _settings(
        monkeypatch,
        org_providers=json.dumps({_ORG_ID: ["xai/grok-voice-think-fast-1.0"]}),
    )
    token = _mint({"org_id": _ORG_ID})
    hidden, _ = _resolve(settings, f"Bearer {token}")
    assert ("xai", "grok-voice-think-fast-1.0") not in hidden
    assert ("xai", "grok-realtime") not in hidden
