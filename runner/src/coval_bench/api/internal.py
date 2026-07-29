# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Internal (benchmarking-team) access to early-access models.

``EARLY_ACCESS`` models run on the normal schedule but are under embargo:
every data endpoint strips them unless the request presents the internal key
in the ``X-Internal-Key`` header. A missing or wrong key silently yields the
public view — the endpoints stay public either way, so there is nothing to 404.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, Header
from pydantic import SecretStr

from coval_bench.api.deps import get_settings
from coval_bench.config import Settings
from coval_bench.registries import MODEL_REGISTRY, ModelStatus

# Full visibility. Deliberately not spellable in a header: it carries a byte no
# HTTP header value may contain, and a parsed scope keeps only names the registry
# already embargoes, so no request value can widen itself into unrestricted access.
UNRESTRICTED = frozenset({"\x00all"})


def embargoed_providers() -> frozenset[str]:
    """The only provider names a request-supplied scope may name."""
    return frozenset(m.provider for m in MODEL_REGISTRY if m.status is ModelStatus.EARLY_ACCESS)


def hidden_models_for(scope: frozenset[str]) -> frozenset[tuple[str, str]]:
    """The ``(provider, model)`` pairs this caller's responses must not contain.

    ``scope`` names the providers this caller may see under embargo. Providers are
    matched by name, so one caller's access never widens to another's.
    """
    if scope & UNRESTRICTED:
        return frozenset()
    return frozenset(
        (m.provider, m.model)
        for m in MODEL_REGISTRY
        if m.status is ModelStatus.EARLY_ACCESS and m.provider not in scope
    )


def hidden_models() -> frozenset[tuple[str, str]]:
    """The ``(provider, model)`` pairs public API responses must not contain."""
    return hidden_models_for(frozenset())


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


def _matches(presented: str | None, expected: SecretStr | None) -> bool:
    if expected is None or presented is None:
        return False
    return hmac.compare_digest(
        presented.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    )


def early_access_scope(
    internal: bool = Depends(is_internal),
    x_ea_key: str | None = Header(default=None),
    x_ea_scope: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> frozenset[str]:
    """Which embargoed providers this caller may see.

    A caller holding the internal key sees everything. Otherwise a scope is
    honoured only alongside its own key, which is separate from the internal one
    so a narrowed caller can never reach unrestricted access. Names outside the
    embargoed set are dropped rather than rejected, so a malformed header is
    indistinguishable from an absent one.
    """
    if internal:
        return UNRESTRICTED
    if not _matches(x_ea_key, settings.early_access_bff_key):
        return frozenset()
    return frozenset((x_ea_scope or "").split(",")) & embargoed_providers()
