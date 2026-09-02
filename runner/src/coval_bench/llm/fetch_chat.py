# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch LLM chat instruction-following scores from the Coval API.

Ingests each provider's recent completed runs (MODEL_TYPE_CHAT agents) not yet
in the DB, slotted by run create_time. Agent ids, the metric id, and the API
key are read from the environment, never committed.

The structure mirrors ``s2s.fetch_v2v`` but is simpler: no latency metric, no
caller-condition splits, no sample publishing. The single metric is instruction
following, scored identically to the S2S benchmark (YES/NO/UNKNOWN per
conversation, aggregated as a pass rate).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.resources
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import click
import httpx
import structlog

from coval_bench.config import Settings, get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.models import Result, ResultStatus, RunStatus
from coval_bench.db.writer import RunWriter
from coval_bench.registries import METRIC_SPECS, Metric
from coval_bench.registries.benchmarks import Benchmark

logger = structlog.get_logger("coval_bench.llm.fetch_chat")

DATASET_ID = "llm-v1"
WINDOW_FLOOR_SECONDS = 86_400
WINDOW_PAGE_SIZE = 10


@dataclass(frozen=True)
class AgentSpec:
    """One LLM chat provider: the Settings attr holding its Coval agent id + display strings."""

    agent_id_attr: str
    provider: str
    model: str


@dataclass(frozen=True)
class CovalRun:
    """One completed Coval run from the list endpoint's summary view."""

    run_id: str
    create_time: datetime | None


AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        agent_id_attr="coval_llm_openai_agent_id",
        provider="openai",
        model="gpt-4o",
    ),
    AgentSpec(
        agent_id_attr="coval_llm_openai_mini_agent_id",
        provider="openai",
        model="gpt-4o-mini",
    ),
    AgentSpec(
        agent_id_attr="coval_llm_anthropic_agent_id",
        provider="anthropic",
        model="claude-sonnet-4-6",
    ),
    AgentSpec(
        agent_id_attr="coval_llm_google_agent_id",
        provider="google",
        model="gemini-2.5-flash",
    ),
)


def _client(settings: Settings) -> httpx.AsyncClient:
    """Build the Coval API client."""
    key = settings.coval_api_key
    if key is None:
        raise RuntimeError("coval_api_key is not set (Secret Manager in prod, .env locally)")
    return httpx.AsyncClient(
        base_url=settings.coval_api_base,
        headers={"X-API-Key": key.get_secret_value()},
        timeout=30.0,
    )


def _dataset_sha256() -> str:
    """SHA-256 of the packaged LLM manifest (or 'unknown' if not yet created)."""
    try:
        ref = importlib.resources.files("coval_bench.datasets.manifests").joinpath(
            f"{DATASET_ID}.json"
        )
        return hashlib.sha256(ref.read_bytes()).hexdigest()
    except Exception:
        return "unknown"


def _parse_time(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _bucket_start(at: datetime, period_seconds: int) -> datetime:
    """Floor a timestamp to the epoch-anchored fetch grid."""
    epoch = int(at.timestamp())
    return datetime.fromtimestamp(epoch - epoch % period_seconds, tz=UTC)


# ---------------------------------------------------------------------------
# Instruction-following value mapper (identical to S2S)
# ---------------------------------------------------------------------------


class InvalidInstructionVerdict(Exception):
    """A run's instruction metric returned a value outside YES/NO/UNKNOWN."""


def _instruction_verdict(raw: object) -> bool | None:
    """YES -> True, NO -> False, UNKNOWN -> None.

    UNKNOWN is excluded from the pass rate. Anything else violates the contract.
    """
    normalized = raw.strip().upper() if isinstance(raw, str) else raw
    if normalized == "YES":
        return True
    if normalized == "NO":
        return False
    if normalized == "UNKNOWN":
        return None
    raise InvalidInstructionVerdict(f"unexpected instruction verdict: {raw!r}")


def _instruction_value(raw: object) -> tuple[float | None, ResultStatus] | None:
    """YES -> 100.0, NO -> 0.0, UNKNOWN -> no row."""
    verdict = _instruction_verdict(raw)
    if verdict is None:
        return None
    return (100.0 if verdict else 0.0), ResultStatus.SUCCESS


_VALUE_MAPPER: Callable[[object], tuple[float | None, ResultStatus] | None] = _instruction_value


# ---------------------------------------------------------------------------
# Run listing
# ---------------------------------------------------------------------------


async def recent_completed_runs(
    client: httpx.AsyncClient,
    agent_id: str,
    *,
    period_seconds: int,
    test_set_id: str | None = None,
    window_seconds: int | None = None,
    page_size: int = WINDOW_PAGE_SIZE,
    requested_run_ids: frozenset[str] | None = None,
) -> list[CovalRun]:
    """Completed Coval runs for one agent within the ingest window, newest first."""
    window = window_seconds or max(WINDOW_FLOOR_SECONDS, 2 * period_seconds)
    filt = f'status="COMPLETED" AND agent_id="{agent_id}"'
    if test_set_id:
        filt += f' AND test_set_id="{test_set_id}"'
    now = datetime.now(tz=UTC)
    runs: list[CovalRun] = []
    page_token: str | None = None
    found_ids: set[str] = set()
    while True:
        params: dict[str, str | int] = {
            "filter": filt,
            "order_by": "-create_time",
            "page_size": page_size,
        }
        if page_token:
            params["page_token"] = page_token
        resp = await client.get("/runs", params=params)
        resp.raise_for_status()
        payload = cast("dict[str, Any]", resp.json())
        raw = cast("list[dict[str, Any]]", payload.get("runs", []))
        for r in raw:
            run = CovalRun(
                run_id=cast("str", r["run_id"]),
                create_time=_parse_time(r.get("create_time")),
            )
            if run.create_time is not None and (now - run.create_time).total_seconds() > window:
                continue
            runs.append(run)
            found_ids.add(run.run_id)
        if not requested_run_ids or requested_run_ids <= found_ids:
            break
        page_token = cast("str | None", payload.get("next_page_token"))
        if not page_token:
            break
    return runs


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _has_duplicate_ids(values: list[dict[str, Any]]) -> bool:
    ids = [v.get("simulation_output_id") for v in values]
    return len(ids) != len(set(ids))


def _instruction_rows(
    values: list[dict[str, Any]],
    *,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """Map per-conversation instruction-following values to Result rows."""
    rows: list[Result] = []
    for i, v in enumerate(values):
        mapped = _VALUE_MAPPER(v.get("value"))
        if mapped is None:
            continue
        metric_value, status = mapped
        sim_id = v.get("simulation_output_id")
        rows.append(
            Result(
                run_id=run_pk,
                provider=spec.provider,
                model=spec.model,
                benchmark=Benchmark.LLM,
                metric_type=Metric.INSTRUCTION_FOLLOWING,
                metric_units=METRIC_SPECS[Metric.INSTRUCTION_FOLLOWING].units,
                metric_value=metric_value,
                audio_filename=f"{coval_run_id}/{sim_id}" if sim_id else f"{coval_run_id}/{i}",
                status=status,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Single-run ingestion
# ---------------------------------------------------------------------------


async def _ingest_run(
    client: httpx.AsyncClient,
    writer: RunWriter,
    *,
    spec: AgentSpec,
    coval_run: CovalRun,
    instruction_metric_id: str,
    dataset_id: str = DATASET_ID,
    dataset_sha256: str = "",
    period_seconds: int,
) -> RunStatus | None:
    """Ingest one Coval run; None = skipped."""
    run_pk: int | None = None
    try:
        resp = await client.get(f"/runs/{coval_run.run_id}")
        resp.raise_for_status()
        run = cast("dict[str, Any]", resp.json()["run"])
        metrics = cast("dict[str, Any]", (run.get("results") or {}).get("metrics") or {})

        payload = metrics.get(instruction_metric_id)
        if payload is None:
            logger.warning(
                "instruction_metric_absent",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
            )
            return None

        values = cast("list[dict[str, Any]]", payload.get("values", []))
        if _has_duplicate_ids(values):
            logger.warning(
                "duplicate_ids",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
            )
            return None

        # Pre-map to check writability before creating a run row.
        try:
            mapped = [_VALUE_MAPPER(v.get("value")) for v in values]
        except InvalidInstructionVerdict as exc:
            logger.warning(
                "instruction_verdict_invalid",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                error=str(exc),
            )
            return None

        if not any(m is not None for m in mapped):
            logger.info(
                "all_unknown_verdicts",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
            )
            return None

        scheduled_at = _bucket_start(coval_run.create_time or datetime.now(tz=UTC), period_seconds)
        run_row = await writer.start_run(
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256 or _dataset_sha256(),
            scheduled_at=scheduled_at,
        )
        if run_row.id is None:
            raise RuntimeError("start_run returned a run with no id")
        run_pk = run_row.id

        rows = _instruction_rows(
            values,
            run_pk=run_pk,
            coval_run_id=coval_run.run_id,
            spec=spec,
        )

        if rows:
            await writer.record_results(rows)

        logger.info(
            "fetched_conversations",
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            slot=str(scheduled_at),
            instruction=len(rows),
        )

        status = RunStatus.SUCCEEDED
        await writer.finish_run(run_pk, status=status)
        try:
            await writer.refresh_bucket(run_pk, period_seconds=period_seconds)
        except Exception:
            logger.warning("refresh_bucket_failed", provider=spec.provider, exc_info=True)
        return status

    except Exception as exc:
        if run_pk is not None:
            try:
                await writer.finish_run(run_pk, status=RunStatus.FAILED, error=str(exc))
            except Exception:
                logger.warning(
                    "finish_run_failed", provider=spec.provider, run_id=run_pk, exc_info=True
                )
        logger.warning(
            "run_ingest_failed",
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            error=str(exc),
        )
        return RunStatus.FAILED


# ---------------------------------------------------------------------------
# Per-provider scan
# ---------------------------------------------------------------------------


async def _fetch_one_provider(
    client: httpx.AsyncClient,
    writer: RunWriter,
    *,
    spec: AgentSpec,
    agent_id: str,
    instruction_metric_id: str,
    test_set_id: str | None = None,
    period_seconds: int,
    stale_grace_seconds: int,
    only_run_ids: frozenset[str] | None = None,
    window_seconds: int | None = None,
    page_size: int = WINDOW_PAGE_SIZE,
    matched_run_ids: set[str] | None = None,
) -> tuple[RunStatus, int]:
    """Scan the window and ingest every not-yet-ingested run.

    Returns (provider status, runs ingested this tick).
    """
    statuses: list[RunStatus] = []
    try:
        runs = await recent_completed_runs(
            client,
            agent_id,
            period_seconds=period_seconds,
            test_set_id=test_set_id,
            window_seconds=window_seconds,
            page_size=page_size,
            requested_run_ids=only_run_ids,
        )
        data_seen = False
        newest_data_at: datetime | None = None
        for coval_run in runs:
            if only_run_ids is not None and coval_run.run_id not in only_run_ids:
                continue

            already = await writer.coval_metric_ingested(
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                metric_type=Metric.INSTRUCTION_FOLLOWING,
            )
            if already:
                if not data_seen:
                    data_seen, newest_data_at = True, coval_run.create_time
                if matched_run_ids is not None and only_run_ids is not None:
                    matched_run_ids.add(coval_run.run_id)
                logger.info(
                    "run_already_ingested", provider=spec.provider, coval_run_id=coval_run.run_id
                )
                continue

            status = await _ingest_run(
                client,
                writer,
                spec=spec,
                coval_run=coval_run,
                instruction_metric_id=instruction_metric_id,
                period_seconds=period_seconds,
            )
            if status is None:
                continue
            if (
                status is not RunStatus.FAILED
                and matched_run_ids is not None
                and only_run_ids is not None
            ):
                matched_run_ids.add(coval_run.run_id)
            if status is not RunStatus.FAILED and not data_seen:
                data_seen, newest_data_at = True, coval_run.create_time
            statuses.append(status)

        if only_run_ids is not None:
            if not statuses:
                return RunStatus.SUCCEEDED, 0
            if all(s is RunStatus.FAILED for s in statuses):
                return RunStatus.FAILED, len(statuses)
            if all(s is RunStatus.SUCCEEDED for s in statuses):
                return RunStatus.SUCCEEDED, len(statuses)
            return RunStatus.PARTIAL, len(statuses)

        threshold = period_seconds + stale_grace_seconds
        age = (
            None
            if newest_data_at is None
            else (datetime.now(tz=UTC) - newest_data_at).total_seconds()
        )
        stale = not data_seen or age is None or age > threshold
        if stale:
            logger.warning(
                "provider_stale",
                provider=spec.provider,
                newest_data_at=str(newest_data_at),
                threshold_seconds=threshold,
            )
            return RunStatus.FAILED, len(statuses)

        if statuses:
            if all(s is RunStatus.FAILED for s in statuses):
                return RunStatus.FAILED, len(statuses)
            if all(s is RunStatus.SUCCEEDED for s in statuses):
                return RunStatus.SUCCEEDED, len(statuses)
            return RunStatus.PARTIAL, len(statuses)
        return RunStatus.SUCCEEDED, 0
    except Exception as exc:
        logger.warning("provider_fetch_failed", provider=spec.provider, error=str(exc))
        return RunStatus.FAILED, len(statuses)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def fetch_and_write_llm(
    settings: Settings | None = None,
    *,
    only_run_ids: frozenset[str] | None = None,
    window_seconds: int | None = None,
    page_size: int = WINDOW_PAGE_SIZE,
) -> dict[str, RunStatus]:
    """Ingest every provider's recent LLM chat runs; return per-provider status."""
    settings = settings or get_settings()

    instruction_metric_id = settings.coval_llm_instruction_metric_id
    if not instruction_metric_id:
        raise RuntimeError("coval_llm_instruction_metric_id is not set")
    test_set_id = settings.coval_llm_test_set_id
    if not test_set_id:
        raise RuntimeError("coval_llm_test_set_id is not set")

    async with _client(settings) as client, lifespan_pool(settings) as pool:
        writer = RunWriter(pool)
        statuses: dict[str, RunStatus] = {}
        total_ingested = 0
        matched_run_ids: set[str] = set()
        for spec in AGENTS:
            agent_id = getattr(settings, spec.agent_id_attr)
            if not agent_id:
                logger.warning("agent_id_unset", provider=spec.provider, attr=spec.agent_id_attr)
                continue
            statuses[f"{spec.provider}:{spec.model}"], ingested = await _fetch_one_provider(
                client,
                writer,
                spec=spec,
                agent_id=agent_id,
                instruction_metric_id=instruction_metric_id,
                test_set_id=test_set_id,
                period_seconds=settings.llm_fetch_period_seconds,
                stale_grace_seconds=settings.llm_stale_grace_seconds,
                only_run_ids=only_run_ids,
                window_seconds=window_seconds,
                page_size=page_size,
                matched_run_ids=matched_run_ids,
            )
            total_ingested += ingested

        if total_ingested:
            try:
                await writer.refresh_stats_matviews()
            except Exception:
                logger.warning("refresh_stats_matviews_failed", exc_info=True)

        if only_run_ids is not None and (unmatched := only_run_ids - matched_run_ids):
            logger.error("backfill_runs_not_found", coval_run_ids=sorted(unmatched))
            raise RuntimeError(
                f"targeted backfill did not recover runs: {', '.join(sorted(unmatched))}"
            )

        logger.info(
            "llm_fetch_done",
            statuses={p: str(s) for p, s in statuses.items()},
            ingested=total_ingested,
        )
        return statuses


@click.command(name="fetch-llm")
@click.option(
    "--coval-run-id",
    "coval_run_ids",
    multiple=True,
    help="Ingest only these Coval runs. Repeatable.",
)
@click.option(
    "--window-hours",
    type=click.IntRange(min=1),
    default=720,
    show_default=True,
    help="How far back --coval-run-id searches. Ignored without it.",
)
@click.option(
    "--page-size",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Runs listed per agent while searching for --coval-run-id. Ignored without it.",
)
def fetch_llm(coval_run_ids: tuple[str, ...], window_hours: int, page_size: int) -> None:
    """Fetch LLM chat instruction-following scores from Coval (scheduled Cloud Run Job).

    With --coval-run-id it becomes a targeted backfill instead.
    """
    from coval_bench.logging import configure_logging, log_run_failed, log_run_partial

    settings = get_settings()
    configure_logging(level=settings.log_level)
    try:
        only_run_ids = frozenset(coval_run_ids) or None
        statuses = asyncio.run(
            fetch_and_write_llm(
                settings,
                only_run_ids=only_run_ids,
                window_seconds=window_hours * 3600 if only_run_ids else None,
                page_size=page_size if only_run_ids else WINDOW_PAGE_SIZE,
            )
        )
    except Exception as exc:
        log_run_failed(str(exc), exc)
        raise

    failed = [p for p, s in statuses.items() if s is RunStatus.FAILED]
    if not statuses or all(s is RunStatus.FAILED for s in statuses.values()):
        if statuses:
            log_run_failed(f"llm fetch failed for all providers: {', '.join(failed)}")
        else:
            log_run_failed("llm fetch ran no providers (none configured)")
        raise click.ClickException("llm fetch failed for all providers")
    if failed:
        log_run_partial(f"llm fetch has no fresh data from: {', '.join(failed)}")


if __name__ == "__main__":
    fetch_llm()
