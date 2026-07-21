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
from collections.abc import Callable

from fastapi import Depends, Header

from coval_bench.api.deps import get_settings
from coval_bench.config import Settings
from coval_bench.registries import MODEL_REGISTRY, STEALTH_PROVIDER, ModelStatus


def hidden_registry_models() -> frozenset[tuple[str, str]]:
    """The registry ``(provider, model)`` pairs public API responses must not contain."""
    return frozenset(
        (m.provider, m.model) for m in MODEL_REGISTRY if m.status is ModelStatus.EARLY_ACCESS
    )


def hidden_predicate(internal: bool) -> Callable[[str, str], bool]:
    """Per-request row filter: True when this caller must not see (provider, model)."""
    if internal:
        return lambda provider, model: False
    hidden = hidden_registry_models()
    return lambda provider, model: provider == STEALTH_PROVIDER or (provider, model) in hidden


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
