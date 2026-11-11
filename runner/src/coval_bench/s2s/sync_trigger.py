# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Start the private s2s-sync job, so the API never holds a key that writes to Coval."""

from __future__ import annotations

import httpx
import structlog

logger = structlog.get_logger("coval_bench.s2s.sync_trigger")

RUN_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
START_TIMEOUT_S = 10.0


def start_s2s_sync(job: str) -> bool:
    """Ask Cloud Run to start ``job`` (a full job resource name); False if it could not."""
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(scopes=[RUN_SCOPE])
        credentials.refresh(google.auth.transport.requests.Request())  # type: ignore[no-untyped-call]
        response = httpx.post(
            f"https://run.googleapis.com/v2/{job}:run",
            headers={"Authorization": f"Bearer {credentials.token}"},
            json={},
            timeout=START_TIMEOUT_S,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("s2s_sync_start_failed", job=job, error=type(exc).__name__)
        return False
    logger.info("s2s_sync_started", job=job)
    return True
