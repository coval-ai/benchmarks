# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Coval-side failure cards, posted to the alerts topic the Slack notifier reads."""

from __future__ import annotations

import base64
import json
import os
import urllib.parse

import httpx
import structlog

logger = structlog.get_logger("coval_bench.alerts")

PUBLISH_TIMEOUT_S = 5.0
PUBSUB_SCOPE = "https://www.googleapis.com/auth/pubsub"


def coval_error_text(exc: BaseException) -> str:
    """A Coval call's failure without its URL or body: the error class and any HTTP status."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{type(exc).__name__} {exc.response.status_code}"
    return type(exc).__name__


def _logs_url(topic: str) -> str | None:
    execution = os.environ.get("CLOUD_RUN_EXECUTION")
    project = topic.split("/")[1] if topic.startswith("projects/") else ""
    if not execution or not project:
        return None
    query = urllib.parse.quote(f'labels."run.googleapis.com/execution_name"="{execution}"')
    return f"https://console.cloud.google.com/logs/query;query={query}?project={project}"


def publish_coval_failure(
    topic: str | None,
    *,
    job: str,
    failure: str,
    error: str,
    scenario: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Post one Coval-side card; does nothing without a topic and never raises."""
    if not topic:
        return
    payload = {
        "source": "coval",
        "job": job,
        "failure": failure,
        "error": error,
        "scenario": scenario,
        "provider": provider,
        "model": model,
        "logs_url": _logs_url(topic),
    }
    message = {k: v for k, v in payload.items() if v}
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(scopes=[PUBSUB_SCOPE])
        credentials.refresh(google.auth.transport.requests.Request())  # type: ignore[no-untyped-call]
        data = base64.b64encode(json.dumps(message).encode()).decode()
        response = httpx.post(
            f"https://pubsub.googleapis.com/v1/{topic}:publish",
            headers={"Authorization": f"Bearer {credentials.token}"},
            json={"messages": [{"data": data}]},
            timeout=PUBLISH_TIMEOUT_S,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("coval_alert_publish_failed", job=job, error=type(exc).__name__)
