# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Starting the s2s-sync job from the API."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from coval_bench.s2s import sync_trigger


class _Credentials:
    token = "fake"  # noqa: S105

    def refresh(self, request: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("google.auth.default", lambda scopes: (_Credentials(), "p"))


def test_starts_the_named_job(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, json={"name": "op"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _post)
    assert sync_trigger.start_s2s_sync("projects/p/locations/r/jobs/s2s-sync")
    assert calls == ["https://run.googleapis.com/v2/projects/p/locations/r/jobs/s2s-sync:run"]


def test_a_refused_start_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def _post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(403, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _post)
    assert not sync_trigger.start_s2s_sync("projects/p/locations/r/jobs/s2s-sync")
