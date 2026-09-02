# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/leaderboard — aggregated benchmark leaderboard.

Metric/benchmark compatibility:
- WER  + STT
- TTFT + STT
- TTFS + STT
- TTFA + TTS
- V2V  + S2S
- TTFT + LLM

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

from coval_bench.api.common import (
    WINDOW_VIEWS,
    BenchmarkLiteral,
    WindowLiteral,
    has_enough_samples,
)
from coval_bench.api.deps import capture_api_event, get_pool, get_posthog
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import LeaderboardEntry, LeaderboardResponse
from coval_bench.config import DATASET_ALL
from coval_bench.registries import is_metric_excluded
from coval_bench.s2s.conditions import DATASET_ID_DENTAL, DATASET_ID_LLM_DENTAL

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
    ("TTFT", "LLM"),
}

# The headline S2S board is the clean dental condition, which every agent now runs.
# The multi-turn set is frozen rather than retired: no agent writes to it, but it
# stays reachable through the aggregates ``dataset`` param, as does ``__all__`` for
# callers that want every S2S condition pooled. LLM runs the same dental scenario
# over text.
_PRIMARY_DATASET_BY_BENCHMARK = {"S2S": DATASET_ID_DENTAL, "LLM": DATASET_ID_LLM_DENTAL}

_MV_SQL_TEMPLATE = """
    SELECT provider, model,
           avg_value AS avg,
           p50,
           p95,
           sample_count AS n
    FROM {view}
    WHERE metric_type = %(metric)s
      AND benchmark = %(benchmark)s
      AND dataset_id = %(dataset)s
    ORDER BY avg_value ASC
"""


@router.get("/leaderboard", response_model=LeaderboardResponse)
@limiter.limit("60/minute")
async def get_leaderboard(
    request: Request,  # required by slowapi
    metric: MetricLiteral = Query(...),
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> LeaderboardResponse:
    """Return leaderboard entries sorted ascending by average metric value.

    Args:
        metric: One of WER, TTFA, TTFT, TTFS.
        benchmark: One of STT, TTS.
        window: Time window — each is served by its materialized view.

    Returns:
        ``{"metric": ..., "window": ..., "entries": [LeaderboardEntry, ...]}``

    Raises:
        400: If the metric/benchmark combination is incompatible.
    """
    if (metric, benchmark) not in _VALID_COMBOS:
        raise HTTPException(
            400,
            f"metric={metric!r} is not compatible with benchmark={benchmark!r}. "
            f"Valid combinations: WER+STT, TTFT+STT, TTFS+STT, TTFA+TTS, V2V+S2S, TTFT+LLM.",
        )

    params: dict[str, Any] = {
        "metric": metric,
        "benchmark": benchmark,
        "dataset": _PRIMARY_DATASET_BY_BENCHMARK.get(benchmark, DATASET_ALL),
    }
    sql = _MV_SQL_TEMPLATE.format(view=WINDOW_VIEWS[window])

    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(sql, params)
        entry_rows = await rows.fetchall()

    entries = [
        LeaderboardEntry.model_validate(r)
        for r in entry_rows
        if (r["provider"], r["model"]) not in hidden
        and not is_metric_excluded(r["provider"], r["model"], metric)
    ]
    entries = [
        e
        if has_enough_samples(benchmark, e.n)
        else e.model_copy(update={"insufficient_samples": True})
        for e in entries
    ]
    # The view ranks by value alone, so a model that scored once with a fast time
    # would lead the board. Thin entries sink below every ranked one; ORDER BY
    # already sorted within each group, and a stable sort preserves it.
    entries.sort(key=lambda e: e.insufficient_samples)
    capture_api_event(
        posthog_client,
        "leaderboard_queried",
        {
            "metric": metric,
            "benchmark": benchmark,
            "window": window,
            "entry_count": len(entries),
            "$process_person_profile": False,
        },
    )
    return LeaderboardResponse(metric=metric, window=window, entries=entries)
