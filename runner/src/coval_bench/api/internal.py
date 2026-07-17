# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Internal (benchmarking-team) access to early-access models.

``EARLY_ACCESS`` models are benchmarked on the normal schedule but are under
embargo: their existence and results must not reach the public API surface.
Every data endpoint strips them from its response unless the request presents
the internal API key in the ``X-Internal-Key`` header, in which case the full
data set is served.

An unset ``internal_api_key`` or a wrong key silently yields the public view —
the endpoints stay public either way, so there is nothing to 404.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, Header

from coval_bench.api.deps import get_settings
from coval_bench.config import Settings
from coval_bench.registries import MODEL_REGISTRY, ModelStatus


def hidden_models() -> frozenset[tuple[str, str]]:
    """The ``(provider, model)`` pairs public API responses must not contain."""
    return frozenset(
        (m.provider, m.model) for m in MODEL_REGISTRY if m.status is ModelStatus.EARLY_ACCESS
    )


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
