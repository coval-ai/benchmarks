# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/leaderboard — aggregated benchmark leaderboard.

Metric/benchmark compatibility:
- WER  + STT
- TTFT + STT
- TTFA + TTS

Every window queries its materialized view (``benchmarks_v2.results_24h``/
``results_7d``/``results_30d``), refreshed by the runner at the end of each
benchmark run — read-only here.
"""

from __future__ import annotations

from typing import Any, Literal

import psycopg.rows
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.common import WINDOW_VIEWS, BenchmarkLiteral, WindowLiteral
from coval_bench.api.deps import capture_api_event, get_pool, get_posthog
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import LeaderboardEntry, LeaderboardResponse
from coval_bench.registries import is_metric_excluded

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["leaderboard"])

MetricLiteral = Literal["WER", "TTFA", "TTFT", "TTFS", "V2V"]

# Valid (metric, benchmark) combinations — lower is better for all.
_VALID_COMBOS: set[tuple[str, str]] = {
    ("WER", "STT"),
    ("TTFT", "STT"),
    ("TTFS", "STT"),
    ("TTFA", "TTS"),
    ("V2V", "S2S"),
}

_MV_SQL_TEMPLATE = """
    SELECT provider, model,
           avg_value AS avg,
           p50,
           p95,
           sample_count AS n
    FROM {view}
    WHERE metric_type = %(metric)s
      AND benchmark = %(benchmark)s{dataset_clause}
    ORDER BY avg_value ASC
"""

_DATASET_CLAUSE = "\n      AND dataset_id = %(dataset)s"


@router.get("/leaderboard", response_model=LeaderboardResponse)
@limiter.limit("60/minute")
async def get_leaderboard(
    request: Request,  # required by slowapi
    metric: MetricLiteral = Query(...),
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    dataset: str | None = Query(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> LeaderboardResponse:
    """Return leaderboard entries sorted ascending by average metric value.

    The matviews aggregate per dataset, so when several datasets of one
    benchmark ran inside the window, a model appears once per dataset unless
    ``dataset`` narrows the board to one.

    Args:
        metric: One of WER, TTFA, TTFT, TTFS.
        benchmark: One of STT, TTS.
        window: Time window — each is served by its materialized view.
        dataset: Optional dataset id (e.g. stt-v2) to filter on.

    Returns:
        ``{"metric": ..., "window": ..., "entries": [LeaderboardEntry, ...]}``

    Raises:
        400: If the metric/benchmark combination is incompatible.
    """
    if (metric, benchmark) not in _VALID_COMBOS:
        raise HTTPException(
            400,
            f"metric={metric!r} is not compatible with benchmark={benchmark!r}. "
            f"Valid combinations: WER+STT, TTFT+STT, TTFS+STT, TTFA+TTS, V2V+S2S.",
        )

    params: dict[str, Any] = {"metric": metric, "benchmark": benchmark}
    dataset_clause = ""
    if dataset is not None:
        dataset_clause = _DATASET_CLAUSE
        params["dataset"] = dataset
    sql = _MV_SQL_TEMPLATE.format(view=WINDOW_VIEWS[window], dataset_clause=dataset_clause)

    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(sql, params)
        entry_rows = await rows.fetchall()

    entries = [
        LeaderboardEntry.model_validate(r)
        for r in entry_rows
        if not is_metric_excluded(r["provider"], r["model"], metric)
    ]
    capture_api_event(
        posthog_client,
        "leaderboard_queried",
        {
            "metric": metric,
            "benchmark": benchmark,
            "window": window,
            "dataset": dataset,
            "entry_count": len(entries),
            "$process_person_profile": False,
        },
    )
    return LeaderboardResponse(metric=metric, window=window, entries=entries)
