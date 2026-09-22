# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation-level reads from normalized benchmark storage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

import psycopg.rows
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr
from starlette.requests import Request

from coval_bench.api.common import BenchmarkLiteral, WindowLiteral
from coval_bench.api.deps import get_pool, get_settings
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.results_cursor import CursorError, CursorKeyError, validate_key
from coval_bench.api.results_cursor import decode as decode_cursor
from coval_bench.api.results_cursor import encode as encode_cursor
from coval_bench.api.schemas import ResultsV2Response, ResultV2Out
from coval_bench.config import Settings

router = APIRouter(tags=["results"])

_CURSOR_VERSION = 1
_RUN_STATUSES = frozenset(("succeeded", "partial"))
_WINDOWS = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}
_DefaultEvaluationVariant = Literal["default"]
_SucceededEvaluationStatus = Literal["succeeded"]
_AllowedRunStatus = Literal["succeeded", "partial"]


def _utc(value: datetime | None, name: str) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(400, f"{name} must include a timezone")
    return value.astimezone(UTC) if value is not None else None


def _fingerprint(filters: dict[str, Any]) -> str:
    encoded = json.dumps(filters, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cursor_payload(
    token: dict[str, object],
) -> tuple[datetime | None, datetime | None, datetime, UUID, str]:
    required = {"v", "fingerprint", "since", "until", "anchor_time", "anchor_id"}
    if set(token) != required or type(token.get("v")) is not int or token["v"] != _CURSOR_VERSION:
        raise HTTPException(400, "invalid cursor")
    fingerprint = token.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != hashlib.sha256().digest_size * 2
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise HTTPException(400, "invalid cursor")
    try:
        since_raw, until_raw = token["since"], token["until"]
        anchor_time_raw, anchor_id_raw = token["anchor_time"], token["anchor_id"]
        if (since_raw is not None and not isinstance(since_raw, str)) or (
            until_raw is not None and not isinstance(until_raw, str)
        ):
            raise ValueError("cursor bounds must be strings or null")
        if not isinstance(anchor_time_raw, str) or not isinstance(anchor_id_raw, str):
            raise ValueError("cursor anchor has invalid type")
        since = datetime.fromisoformat(since_raw) if since_raw is not None else None
        until = datetime.fromisoformat(until_raw) if until_raw is not None else None
        anchor_time = datetime.fromisoformat(anchor_time_raw)
        anchor_id = UUID(anchor_id_raw)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(400, "invalid cursor") from exc
    normalized_since = _utc(since, "cursor since")
    normalized_until = _utc(until, "cursor until")
    normalized_anchor_time = _utc(anchor_time, "cursor anchor")
    if normalized_anchor_time is None or (
        normalized_since is not None
        and normalized_until is not None
        and normalized_since >= normalized_until
    ):
        raise HTTPException(400, "invalid cursor")
    return normalized_since, normalized_until, normalized_anchor_time, anchor_id, fingerprint


def _unsupported_status(name: str) -> HTTPException:
    return HTTPException(422, f"{name} is restricted for normalized results")


def _require_cursor_key(settings: Settings = Depends(get_settings)) -> SecretStr:
    try:
        return validate_key(settings.results_cursor_key)
    except CursorKeyError as exc:
        raise HTTPException(503, "results pagination is unavailable") from exc


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
    evaluation_variant: _DefaultEvaluationVariant = Query(
        default="default", description="Only default-variant evaluations are public."
    ),
    evaluation_status: _SucceededEvaluationStatus = Query(
        default="succeeded", description="Only successful evaluations are public."
    ),
    run_status: _AllowedRunStatus | None = Query(
        default=None, description="Parent run state; omitted includes succeeded and partial."
    ),
    include_components: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=1000),
    window: WindowLiteral | None = Query(
        default=None,
        description="Relative capture window; defaults to 7d only without run_id or bounds.",
    ),
    since: datetime | None = Query(
        default=None, description="Inclusive timezone-aware capture bound; excludes window."
    ),
    until: datetime | None = Query(
        default=None, description="Exclusive timezone-aware capture bound; excludes window."
    ),
    cursor: str | None = Query(default=None, description="Opaque token from the prior page."),
    cursor_key: SecretStr = Depends(_require_cursor_key),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> ResultsV2Response:
    """Return one successful primary evaluation per normalized result."""
    del request
    since = _utc(since, "since")
    until = _utc(until, "until")
    if since is not None and until is not None and since >= until:
        raise HTTPException(400, "since must be before until")
    if window is not None and (since is not None or until is not None):
        raise HTTPException(400, "window cannot be combined with since/until")
    if evaluation_status != "succeeded":
        raise _unsupported_status("evaluation_status")
    if evaluation_variant != "default":
        raise _unsupported_status("evaluation_variant")
    parent_statuses = ("succeeded", "partial") if run_status is None else (run_status,)
    if any(status not in _RUN_STATUSES for status in parent_statuses):
        raise _unsupported_status("run_status")

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
    anchor_time: datetime | None = None
    anchor_id: UUID | None = None
    expected: str | None = None
    try:
        if cursor is not None:
            decoded = decode_cursor(cursor, cursor_key)
            frozen_since, frozen_until, anchor_time, anchor_id, expected = _cursor_payload(decoded)
            since, until = frozen_since, frozen_until
    except CursorKeyError as exc:
        raise HTTPException(503, "results pagination is unavailable") from exc
    except CursorError as exc:
        raise HTTPException(400, "invalid cursor") from exc

    filter_payload = {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "benchmark": benchmark,
        "dataset": dataset,
        "metric_type": metric_type,
        "metric_version": metric_version,
        "evaluation_variant": "default",
        "evaluation_status": "succeeded",
        "run_status": parent_statuses,
        "time": requested_time,
        "hidden": sorted(hidden),
    }
    fingerprint = _fingerprint(filter_payload)
    if cursor is not None and expected != fingerprint:
        raise HTTPException(400, "cursor does not match request filters or visibility")

    conditions = [
        "e.status = 'succeeded'",
        "e.evaluation_variant = 'default'",
        "r.status = ANY(%(run_statuses)s)",
    ]
    params: dict[str, Any] = {
        "run_statuses": list(parent_statuses),
        "page_limit": limit + 1,
        "include_components": include_components,
    }
    if run_id is not None:
        conditions.append("o.run_id = %(run_id)s")
        params["run_id"] = run_id
    for name, column, value in (
        ("provider", "o.provider", provider),
        ("model", "o.model", model),
        ("benchmark", "o.benchmark", benchmark),
        ("dataset", "o.dataset_id", dataset),
    ):
        if value is not None:
            conditions.append(f"{column} = %({name})s")
            params[name] = value
    if metric_type is not None:
        conditions.append("m.code = %(metric_type)s")
        params["metric_type"] = metric_type
    if metric_version is not None:
        conditions.append("e.metric_version = %(metric_version)s")
        params["metric_version"] = metric_version
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
          SELECT e.id AS evaluation_id, o.captured_at, m.code AS metric_type,
                 e.metric_version, primary_value.value, primary_value.unit,
                 o.provider, o.model, o.voice, o.benchmark, o.dataset_id,
                 r.status AS run_status
          FROM benchmarks_v2.metric_evaluations e
          JOIN benchmarks_v2.benchmark_observations o ON o.id = e.observation_id
          JOIN benchmarks_v2.runs r ON r.id = o.run_id
          JOIN benchmarks_v2.metrics m ON m.id = e.metric_id
          JOIN benchmarks_v2.metric_values primary_value
            ON primary_value.metric_evaluation_id = e.id
           AND primary_value.value_role = 'primary'
          WHERE {" AND ".join(conditions)}
          ORDER BY o.captured_at DESC, e.id DESC
          LIMIT %(page_limit)s
        )
        SELECT p.evaluation_id AS _cursor_evaluation_id, p.metric_type, p.metric_version,
               p.value, p.unit, p.provider, p.model, p.voice, p.benchmark,
               p.dataset_id, p.captured_at, components.components
        FROM evaluation_page p
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
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        result = await conn.execute(page_sql, params)
        rows = await result.fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    response_rows: list[ResultV2Out] = []
    for row in rows:
        payload: dict[str, Any] = {
            "metric_type": row["metric_type"],
            "metric_version": row["metric_version"],
            "value": row["value"],
            "unit": row["unit"],
            "provider": row["provider"],
            "model": row["model"],
            "voice": row["voice"],
            "benchmark": row["benchmark"],
            "dataset_id": row["dataset_id"],
            "captured_at": row["captured_at"],
        }
        if include_components:
            payload["components"] = row["components"]
        response_rows.append(ResultV2Out.model_validate(payload))
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        try:
            next_cursor = encode_cursor(
                {
                    "v": _CURSOR_VERSION,
                    "fingerprint": fingerprint,
                    "since": since.isoformat() if since else None,
                    "until": until.isoformat() if until else None,
                    "anchor_time": last["captured_at"].isoformat(),
                    "anchor_id": str(last["_cursor_evaluation_id"]),
                },
                cursor_key,
            )
        except (CursorError, CursorKeyError) as exc:
            raise HTTPException(503, "results pagination is unavailable") from exc
    return ResultsV2Response(results=response_rows, next_cursor=next_cursor)
