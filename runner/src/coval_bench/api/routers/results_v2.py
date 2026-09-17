# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation-level reads from benchmark storage."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

import psycopg.rows
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.common import BenchmarkLiteral, WindowLiteral
from coval_bench.api.deps import get_pool
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ResultsV2Response

router = APIRouter(tags=["results"])
_CURSOR_VERSION = 1
_MAX_CURSOR_LENGTH = 8192
_RUN_STATUSES = frozenset(("running", "succeeded", "partial", "failed"))
_EVALUATION_STATUSES = frozenset(("queued", "running", "succeeded", "failed"))
_WINDOWS = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}
EvaluationStatusLiteral = Literal["queued", "running", "succeeded", "failed", "all"]
RunStatusLiteral = Literal["running", "succeeded", "partial", "failed", "all"]


def _utc(value: datetime | None, name: str) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(400, f"{name} must include a timezone")
    return value.astimezone(UTC) if value is not None else None


def _decode_cursor(value: str) -> dict[str, Any]:
    if len(value) > _MAX_CURSOR_LENGTH:
        raise HTTPException(400, "cursor is too large")
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii") + b"=" * (-len(value) % 4))
        decoded = json.loads(raw)
    except (binascii.Error, ValueError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise HTTPException(400, "invalid cursor") from exc
    if not isinstance(decoded, dict) or decoded.get("v") != _CURSOR_VERSION:
        raise HTTPException(400, "invalid cursor")
    if set(decoded) != {"v", "fingerprint", "since", "until", "anchor_time", "anchor_id"}:
        raise HTTPException(400, "invalid cursor")
    if (
        type(decoded.get("v")) is not int
        or not isinstance(decoded.get("fingerprint"), str)
        or len(decoded["fingerprint"]) != hashlib.sha256().digest_size * 2
        or any(c not in "0123456789abcdef" for c in decoded["fingerprint"])
        or not isinstance(decoded.get("anchor_time"), str)
        or not isinstance(decoded.get("anchor_id"), str)
        or not isinstance(decoded.get("since"), (str, type(None)))
        or not isinstance(decoded.get("until"), (str, type(None)))
    ):
        raise HTTPException(400, "invalid cursor")
    return decoded


def _encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _fingerprint(filters: dict[str, Any]) -> str:
    encoded = json.dumps(filters, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _status_filter(
    value: str | None, allowed: frozenset[str], name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    if value is None:
        return default
    if value == "all":
        return tuple(sorted(allowed))
    if value not in allowed:
        raise HTTPException(400, f"invalid {name}")
    return (value,)


def _parse_cursor(
    token: dict[str, Any],
) -> tuple[datetime | None, datetime | None, datetime, UUID, str]:
    try:
        frozen_since = (
            datetime.fromisoformat(token["since"]) if token["since"] is not None else None
        )
        frozen_until = (
            datetime.fromisoformat(token["until"]) if token["until"] is not None else None
        )
        anchor_time = datetime.fromisoformat(token["anchor_time"])
        anchor_id = UUID(token["anchor_id"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(400, "invalid cursor") from exc
    frozen_since = _utc(frozen_since, "cursor since")
    frozen_until = _utc(frozen_until, "cursor until")
    normalized_anchor_time = _utc(anchor_time, "cursor anchor")
    if normalized_anchor_time is None or (
        frozen_since is not None and frozen_until is not None and frozen_since >= frozen_until
    ):
        raise HTTPException(400, "invalid cursor")
    return frozen_since, frozen_until, normalized_anchor_time, anchor_id, token["fingerprint"]


@router.get(
    "/results",
    response_model=ResultsV2Response,
    response_model_exclude_unset=True,
)
@limiter.limit("60/minute")
async def list_results(
    request: Request,
    run_id: int | None = Query(default=None, gt=0),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    benchmark: BenchmarkLiteral | None = Query(default=None),
    dataset: str | None = Query(default=None),
    metric_type: str | None = Query(default=None),
    metric_version: str | None = Query(
        default=None, description="Exact metric version; omitted returns all versions."
    ),
    evaluation_variant: str | None = Query(
        default=None, description="Exact evaluation variant; omitted returns all variants."
    ),
    evaluation_status: EvaluationStatusLiteral = Query(
        default="succeeded",
        description="Evaluation state to include. Defaults to succeeded; use all for every state.",
    ),
    run_status: RunStatusLiteral | None = Query(
        default=None,
        description=(
            "Parent run state to include. Omitted defaults to succeeded and partial; use all "
            "for every state."
        ),
    ),
    include_components: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=1000),
    window: WindowLiteral | None = Query(
        default=None,
        description=(
            "Relative capture window; defaults to 7d only when no run_id or bounds are supplied."
        ),
    ),
    since: datetime | None = Query(
        default=None,
        description="Inclusive timezone-aware capture bound; mutually exclusive with window.",
    ),
    until: datetime | None = Query(
        default=None,
        description="Exclusive timezone-aware capture bound; mutually exclusive with window.",
    ),
    cursor: str | None = Query(
        default=None, description="Opaque continuation token from the prior page."
    ),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> ResultsV2Response:
    """Return one item per metric evaluation, newest observation first."""
    since = _utc(since, "since")
    until = _utc(until, "until")
    if since is not None and until is not None and since >= until:
        raise HTTPException(400, "since must be before until")
    if window is not None and (since is not None or until is not None):
        raise HTTPException(400, "window cannot be combined with since/until")
    eval_statuses = _status_filter(
        evaluation_status, _EVALUATION_STATUSES, "evaluation_status", ("succeeded",)
    )
    parent_statuses = _status_filter(
        run_status, _RUN_STATUSES, "run_status", ("succeeded", "partial")
    )

    requested_time = {
        "window": window,
        "since": since.isoformat() if since is not None else None,
        "until": until.isoformat() if until is not None else None,
    }
    now = datetime.now(UTC)
    if cursor is None:
        if run_id is None and window is None and since is None and until is None:
            since, until = now - _WINDOWS["7d"], now
        elif window is not None:
            since, until = now - _WINDOWS[window], now
    else:
        token = _decode_cursor(cursor)
        since, until, anchor_time, anchor_id, expected = _parse_cursor(token)

    filter_payload = {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "benchmark": benchmark,
        "dataset": dataset,
        "metric_type": metric_type,
        "metric_version": metric_version,
        "evaluation_variant": evaluation_variant,
        "evaluation_status": eval_statuses,
        "run_status": parent_statuses,
        "time": requested_time,
        "hidden": sorted(hidden),
    }
    fingerprint = _fingerprint(filter_payload)
    if cursor is not None and expected != fingerprint:
        raise HTTPException(400, "cursor does not match request filters or visibility")

    conditions = ["e.status = ANY(%(evaluation_statuses)s)", "r.status = ANY(%(run_statuses)s)"]
    params: dict[str, Any] = {
        "evaluation_statuses": list(eval_statuses),
        "run_statuses": list(parent_statuses),
        "page_limit": limit + 1,
    }
    if run_id is not None:
        conditions.append("o.run_id = %(run_id)s")
        params["run_id"] = run_id
    for name, column in (
        ("provider", "o.provider"),
        ("model", "o.model"),
        ("benchmark", "o.benchmark"),
        ("dataset", "o.dataset_id"),
    ):
        value = locals()[name]
        if value is not None:
            conditions.append(f"{column} = %({name})s")
            params[name] = value
    if metric_type is not None:
        conditions.append("m.code = %(metric_type)s")
        params["metric_type"] = metric_type
    if metric_version is not None:
        conditions.append("e.metric_version = %(metric_version)s")
        params["metric_version"] = metric_version
    if evaluation_variant is not None:
        conditions.append("e.evaluation_variant = %(evaluation_variant)s")
        params["evaluation_variant"] = evaluation_variant
    if since is not None:
        conditions.append("o.captured_at >= %(since)s")
        params["since"] = since
    if until is not None:
        conditions.append("o.captured_at < %(until)s")
        params["until"] = until
    for index, (hidden_provider, hidden_model) in enumerate(sorted(hidden)):
        conditions.append(
            f"NOT (o.provider = %(hidden_provider_{index})s AND o.model = %(hidden_model_{index})s)"
        )
        params[f"hidden_provider_{index}"] = hidden_provider
        params[f"hidden_model_{index}"] = hidden_model
    if cursor is not None:
        conditions.append("o.captured_at <= %(anchor_time)s")
        conditions.append("(o.captured_at, e.id) < (%(anchor_time)s, %(anchor_id)s)")
        params["anchor_time"] = anchor_time
        params["anchor_id"] = anchor_id

    page_sql = f"""
        WITH evaluation_page AS MATERIALIZED (
          SELECT e.id AS evaluation_id, e.observation_id, o.run_id, m.code AS metric_type,
                 e.metric_version, e.evaluation_variant, e.status AS evaluation_status,
                 r.status AS run_status, o.provider, o.model, o.voice, o.benchmark,
                 o.dataset_id, o.sample_id, o.captured_at
          FROM benchmarks_v2.metric_evaluations e
          JOIN benchmarks_v2.benchmark_observations o ON o.id = e.observation_id
          JOIN benchmarks_v2.runs r ON r.id = o.run_id
          JOIN benchmarks_v2.metrics m ON m.id = e.metric_id
          WHERE {" AND ".join(conditions)}
          ORDER BY o.captured_at DESC, e.id DESC
          LIMIT %(page_limit)s
        )
        SELECT p.*, primary_value.value, primary_value.unit,
               components.components
        FROM evaluation_page p
        LEFT JOIN benchmarks_v2.metric_values primary_value
          ON primary_value.metric_evaluation_id = p.evaluation_id
         AND primary_value.value_role = 'primary'
        LEFT JOIN LATERAL (
          SELECT COALESCE(
            jsonb_object_agg(v.value_key, jsonb_build_object('value', v.value, 'unit', v.unit)),
            '{{}}'::jsonb
          ) AS components
          FROM benchmarks_v2.metric_values v
          WHERE v.metric_evaluation_id = p.evaluation_id AND v.value_role = 'component'
        ) components ON %(include_components)s
        ORDER BY p.captured_at DESC, p.evaluation_id DESC
    """  # noqa: S608
    params["include_components"] = include_components
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        result = await conn.execute(page_sql, params)
        rows = await result.fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    response_rows = []
    for row in rows:
        payload = dict(row)
        if include_components:
            payload["components"] = payload.pop("components")
        else:
            payload.pop("components", None)
        response_rows.append(payload)
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = _encode_cursor(
            {
                "v": _CURSOR_VERSION,
                "fingerprint": fingerprint,
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
                "anchor_time": last["captured_at"].isoformat(),
                "anchor_id": str(last["evaluation_id"]),
            }
        )
    return ResultsV2Response(results=response_rows, next_cursor=next_cursor)
