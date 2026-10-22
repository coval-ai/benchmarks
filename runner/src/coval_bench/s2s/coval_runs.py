# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The Coval API client and the completed-run listing one tick scans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import httpx
import structlog

if TYPE_CHECKING:
    from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.s2s.coval_runs")


# Ingest window: how far back one tick looks for not-yet-ingested runs, and
# how many list results that scan reads. Two periods (with a one-day floor)
# so the previous run stays visible while the current one is due, and a run
# resolving late is still picked up.
WINDOW_FLOOR_SECONDS = 86_400
WINDOW_PAGE_SIZE = 10


@dataclass(frozen=True)
class CovalRun:
    """One completed Coval run from the list endpoint's summary view."""

    run_id: str
    create_time: datetime | None
    persona_id: str = ""


def coval_client(settings: Settings) -> httpx.AsyncClient:
    """Build the Coval API client. The key is used only as a header, never logged."""
    key = settings.coval_api_key
    if key is None:
        raise RuntimeError("coval_api_key is not set (Secret Manager in prod, .env locally)")
    return httpx.AsyncClient(
        base_url=settings.coval_api_base,
        headers={"X-API-Key": key.get_secret_value()},
        timeout=30.0,
    )


def parse_time(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def bucket_start(at: datetime, period_seconds: int) -> datetime:
    """Floor a timestamp to the epoch-anchored fetch grid."""
    epoch = int(at.timestamp())
    return datetime.fromtimestamp(epoch - epoch % period_seconds, tz=UTC)


async def recent_completed_runs(
    client: httpx.AsyncClient,
    agent_id: str,
    *,
    period_seconds: int,
    test_set_id: str | None = None,
    window_seconds: int | None = None,
    page_size: int = WINDOW_PAGE_SIZE,
    requested_run_ids: frozenset[str] | None = None,
    workspace_id: str | None = None,
) -> list[CovalRun]:
    """Completed Coval runs for one agent within the ingest window, newest first.

    List returns the summary view newest-first; the API WAF rejects AND
    filters, so filter by agent_id and check status and test set here.
    ``test_set_id`` narrows to one test set so other sims on the same agents
    (e.g. the single-turn set) are not ingested. Runs
    without a parseable create_time are kept: better to ingest with a fetch-time
    slot than to drop data. Pages are followed until one reaches past the window,
    so an agent with several runs a day (one per persona) is never cut off at the
    first page. ``workspace_id`` is required for an agent that does not live in
    the workspace the client's API key defaults to.
    """
    window = window_seconds or max(WINDOW_FLOOR_SECONDS, 2 * period_seconds)
    filt = f'agent_id="{agent_id}"'
    now = datetime.now(tz=UTC)
    runs: list[CovalRun] = []
    page_token: str | None = None
    found_ids: set[str] = set()
    headers = {"X-Coval-Workspace-Id": workspace_id} if workspace_id else None
    while True:
        params: dict[str, str | int] = {
            "filter": filt,
            "order_by": "-create_time",
            "page_size": page_size,
        }
        if page_token:
            params["page_token"] = page_token
        resp = await client.get("/runs", params=params, headers=headers)
        resp.raise_for_status()
        payload = cast("dict[str, Any]", resp.json())
        raw = cast("list[dict[str, Any]]", payload.get("runs", []))
        left_window = False
        for r in raw:
            run = CovalRun(
                run_id=cast("str", r["run_id"]),
                create_time=parse_time(r.get("create_time")),
                persona_id=cast("str", r.get("persona_id") or ""),
            )
            if run.create_time is not None and (now - run.create_time).total_seconds() > window:
                left_window = True
                continue
            if r.get("status") != "COMPLETED":
                continue
            if test_set_id and r.get("test_set_id") != test_set_id:
                continue
            runs.append(run)
            found_ids.add(run.run_id)
        if requested_run_ids:
            if requested_run_ids <= found_ids:
                break
        elif left_window or not raw:
            break
        page_token = cast("str | None", payload.get("next_page_token"))
        if not page_token:
            break
    return runs
