# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
from typing import Any

import google.auth
import httpx
import pytest

from coval_bench import alerts


class _Credentials:
    token = "tok"  # noqa: S105

    def refresh(self, _request: object) -> None:
        return None


def test_coval_error_text_drops_the_url() -> None:
    request = httpx.Request("GET", "https://api.coval.dev/v1/runs?filter=secret")
    response = httpx.Response(403, request=request, text="<html>blocked</html>")
    exc = httpx.HTTPStatusError(
        "403 for https://api.coval.dev/v1/runs?filter=secret", request=request, response=response
    )

    assert alerts.coval_error_text(exc) == "HTTPStatusError 403"
    assert (
        alerts.coval_error_text(httpx.ConnectTimeout("slow", request=request)) == "ConnectTimeout"
    )


def test_publish_does_nothing_without_a_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("should not publish")

    monkeypatch.setattr(httpx, "post", fail)

    alerts.publish_coval_failure("", job="s2s-fetch", failure="api_request", error="x")


def test_publish_sends_one_coval_message(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent["url"] = url
        sent.update(kwargs)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(google.auth, "default", lambda scopes: (_Credentials(), "p"))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("CLOUD_RUN_EXECUTION", "s2s-fetch-abc12")

    alerts.publish_coval_failure(
        "projects/p/topics/alerts",
        job="s2s-fetch",
        failure="api_request",
        error="HTTPStatusError 503",
        scenario="s2s-bank",
        provider="google",
        model="gemini-3.8-live",
    )

    assert sent["url"] == "https://pubsub.googleapis.com/v1/projects/p/topics/alerts:publish"
    assert sent["headers"] == {"Authorization": "Bearer tok"}
    message = json.loads(base64.b64decode(sent["json"]["messages"][0]["data"]))
    assert message["source"] == "coval"
    assert message["provider"] == "google"
    assert message["error"] == "HTTPStatusError 503"
    assert "s2s-fetch-abc12" in message["logs_url"]
    assert message["logs_url"].endswith("?project=p")


def test_publish_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(url: str, **_kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("down", request=httpx.Request("POST", url))

    monkeypatch.setattr(google.auth, "default", lambda scopes: (_Credentials(), "p"))
    monkeypatch.setattr(httpx, "post", post)

    alerts.publish_coval_failure(
        "projects/p/topics/alerts", job="s2s-fetch", failure="api_request", error="x"
    )
