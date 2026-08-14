# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch S2S (voice-to-voice) latency from the Coval API and write per-clip rows.

Ingests each provider's recent completed runs not yet in the DB, slotted by
run create_time, and flags providers with no fresh data. Agent ids, the
metric id, and the API key are read from the environment, never committed.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.resources
import random
from collections.abc import Callable, Mapping
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
from coval_bench.s2s.conditions import (
    DATASET_ID,
    DATASET_ID_MULTITURN,
    DATASET_ID_MULTITURN_NOISY,
    DEFAULT_CONDITION,
    DatasetMetrics,
    condition_for,
)
from coval_bench.s2s.samples import SampleRun, model_labels, publish_tick_sample

logger = structlog.get_logger("coval_bench.s2s.fetch_v2v")

# Ingest window: how far back one tick looks for not-yet-ingested runs, and
# how many list results that scan reads. Two periods (with a one-day floor)
# so the previous run stays visible while the current one is due, and a run
# resolving late is still picked up.
WINDOW_FLOOR_SECONDS = 86_400
WINDOW_PAGE_SIZE = 10


@dataclass(frozen=True)
class AgentSpec:
    """One S2S provider: the Settings attr holding its Coval agent id + display strings."""

    agent_id_attr: str
    provider: str
    model: str


@dataclass(frozen=True)
class CovalRun:
    """One completed Coval run from the list endpoint's summary view."""

    run_id: str
    create_time: datetime | None
    error_status: str | None
    persona_id: str = ""


# Agent ids resolved from Settings; model strings are display labels.
AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(agent_id_attr="coval_s2s_openai_agent_id", provider="openai", model="gpt-realtime"),
    AgentSpec(agent_id_attr="coval_s2s_gemini_agent_id", provider="google", model="gemini-live"),
    AgentSpec(
        agent_id_attr="coval_s2s_xai_agent_id",
        provider="xai",
        model="grok-voice-think-fast-1.0",
    ),
    AgentSpec(
        agent_id_attr="coval_s2s_xai_think_fast_2_agent_id",
        provider="xai",
        model="grok-voice-think-fast-2.0",
    ),
)


def _client(settings: Settings) -> httpx.AsyncClient:
    """Build the Coval API client. The key is used only as a header, never logged."""
    key = settings.coval_api_key
    if key is None:
        raise RuntimeError("coval_api_key is not set (Secret Manager in prod, .env locally)")
    return httpx.AsyncClient(
        base_url=settings.coval_api_base,
        headers={"X-API-Key": key.get_secret_value()},
        timeout=30.0,
    )


def _dataset_sha256() -> str:
    """SHA-256 of the packaged S2S manifest."""
    try:
        ref = importlib.resources.files("coval_bench.datasets.manifests").joinpath(
            f"{DATASET_ID}.json"
        )
        return hashlib.sha256(ref.read_bytes()).hexdigest()
    except Exception:
        logger.warning("dataset_sha256_failed", dataset_id=DATASET_ID, exc_info=True)
        return "unknown"


def _dataset_identity(
    test_set_id: str | None,
    persona_id: str = "",
    noisy_persona_id: str | None = None,
) -> tuple[str, str]:
    """Dataset id + provenance for one run's rows.

    Multi-turn runs are keyed to their Coval test set, not the single-turn SLURP
    manifest, so they get their own dataset id (never pooling with s2s-v1) and
    record the test-set id as provenance. Without a test set (legacy latency-only
    mode) the rows stay under the packaged s2s-v1 manifest.

    The noisy caller shares that test set, so only the persona separates the two
    conditions; its provenance names the persona to keep the rows self-describing.
    """
    if test_set_id:
        if noisy_persona_id and persona_id == noisy_persona_id:
            return DATASET_ID_MULTITURN_NOISY, f"{test_set_id}:{persona_id}"
        return DATASET_ID_MULTITURN, test_set_id
    return DATASET_ID, _dataset_sha256()


def _parse_time(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _error_status(raw: object) -> str | None:
    """Normalize Coval's error_status field; clean runs report the string "SUCCESS"."""
    status = cast("str | None", raw) or None
    return None if status == "SUCCESS" else status


def _bucket_start(at: datetime, period_seconds: int) -> datetime:
    """Floor a timestamp to the epoch-anchored fetch grid."""
    epoch = int(at.timestamp())
    return datetime.fromtimestamp(epoch - epoch % period_seconds, tz=UTC)


async def recent_completed_runs(
    client: httpx.AsyncClient,
    agent_id: str,
    *,
    period_seconds: int,
    test_set_id: str | None = None,
) -> list[CovalRun]:
    """Completed Coval runs for one agent within the ingest window, newest first.

    List returns the summary view newest-first; filter by agent_id + status
    (tags are not filterable). ``test_set_id`` narrows to one test set so other
    sims on the same agents (e.g. the single-turn set) are not ingested. Runs
    without a parseable create_time are kept: better to ingest with a fetch-time
    slot than to drop data.
    """
    window_seconds = max(WINDOW_FLOOR_SECONDS, 2 * period_seconds)
    filt = f'status="COMPLETED" AND agent_id="{agent_id}"'
    if test_set_id:
        filt += f' AND test_set_id="{test_set_id}"'
    resp = await client.get(
        "/runs",
        params={
            "filter": filt,
            "order_by": "-create_time",
            "page_size": WINDOW_PAGE_SIZE,
        },
    )
    resp.raise_for_status()
    raw = cast("list[dict[str, Any]]", resp.json().get("runs", []))
    now = datetime.now(tz=UTC)
    runs: list[CovalRun] = []
    for r in raw:
        run = CovalRun(
            run_id=cast("str", r["run_id"]),
            create_time=_parse_time(r.get("create_time")),
            error_status=_error_status(r.get("error_status")),
            persona_id=cast("str", r.get("persona_id") or ""),
        )
        if run.create_time is not None and (now - run.create_time).total_seconds() > window_seconds:
            continue
        runs.append(run)
    return runs


def _v2v_value(raw: object) -> tuple[float | None, ResultStatus] | None:
    """Seconds to ms; a clip with no numeric value becomes a FAILED row."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return round(float(raw) * 1000, 1), ResultStatus.SUCCESS
    return None, ResultStatus.FAILED


class InvalidInstructionVerdict(Exception):
    """A run's instruction metric returned a value outside YES/NO/UNKNOWN."""


def _instruction_verdict(raw: object) -> bool | None:
    """Classify a binary judge verdict: YES -> True, NO -> False, UNKNOWN -> None.

    UNKNOWN (None) is excluded from the pass rate. A binary judge only ever emits
    canonical YES/NO/UNKNOWN, so anything else violates the contract and raises,
    rather than being silently scored as a miss.
    """
    normalized = raw.strip().upper() if isinstance(raw, str) else raw
    if normalized == "YES":
        return True
    if normalized == "NO":
        return False
    if normalized == "UNKNOWN":
        return None
    raise InvalidInstructionVerdict(f"unexpected instruction verdict: {raw!r}")


def _has_duplicate_ids(values: list[dict[str, Any]]) -> bool:
    """True if two of a metric's conversations share a simulation id."""
    ids = [v.get("simulation_output_id") for v in values]
    return len(ids) != len(set(ids))


def _population_mismatch(
    anchor_values: list[dict[str, Any]], other_values: list[dict[str, Any]]
) -> dict[str, object] | None:
    """None if *other_values* covers the same conversations as the anchor.

    Ids, not counts: equal counts can be different conversations. A mismatch skips
    that metric for the run.
    """
    other_set = {v.get("simulation_output_id") for v in other_values}
    anchor_set = {v.get("simulation_output_id") for v in anchor_values}
    missing = sorted(str(x) for x in anchor_set - other_set)
    extra = sorted(str(x) for x in other_set - anchor_set)
    duplicate = _has_duplicate_ids(other_values)
    if not duplicate and not missing and not extra:
        return None
    return {"missing_ids": missing, "extra_ids": extra, "duplicate_ids": duplicate}


def _instruction_value(raw: object) -> tuple[float | None, ResultStatus] | None:
    """YES -> 100.0, NO -> 0.0, UNKNOWN -> no row, so the mean is YES / (YES + NO).

    Raises InvalidInstructionVerdict on any value outside the contract.
    """
    verdict = _instruction_verdict(raw)
    if verdict is None:
        return None
    return (100.0 if verdict else 0.0), ResultStatus.SUCCESS


# The only per-metric part: one raw Coval value -> (row value, status), or None to
# write no row. A metric is ingestable once it appears here.
_VALUE_MAPPERS: dict[Metric, Callable[[object], tuple[float | None, ResultStatus] | None]] = {
    Metric.V2V: _v2v_value,
    Metric.INSTRUCTION_FOLLOWING: _instruction_value,
}


def _s2s_rows(
    values: list[dict[str, Any]],
    *,
    metric: Metric,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """Map one metric's per-conversation values to Result rows."""
    to_row = _VALUE_MAPPERS[metric]
    rows: list[Result] = []
    for i, v in enumerate(values):
        mapped = to_row(v.get("value"))
        if mapped is None:
            continue
        metric_value, status = mapped
        sim_id = v.get("simulation_output_id")
        rows.append(
            Result(
                run_id=run_pk,
                provider=spec.provider,
                model=spec.model,
                benchmark=Benchmark.S2S,
                metric_type=metric,
                metric_units=METRIC_SPECS[metric].units,
                metric_value=metric_value,
                audio_filename=f"{coval_run_id}/{sim_id}" if sim_id else f"{coval_run_id}/{i}",
                status=status,
            )
        )
    return rows


def _metric_values(metrics: dict[str, Any], metric_id: str | None) -> list[dict[str, Any]] | None:
    """The metric's per-conversation values, or None when it is not on the run."""
    payload = metrics.get(metric_id) if metric_id else None
    if payload is None:
        return None
    return cast("list[dict[str, Any]]", payload.get("values", []))


def _ingestable(condition: DatasetMetrics, metric_ids: Mapping[Metric, str]) -> frozenset[Metric]:
    """The condition's metrics that are configured and have a row builder."""
    return frozenset(m for m in condition.fetched if m in metric_ids and m in _VALUE_MAPPERS)


async def _pending_metrics(
    writer: RunWriter,
    *,
    provider: str,
    coval_run_id: str,
    condition: DatasetMetrics,
    metric_ids: Mapping[Metric, str],
) -> frozenset[Metric]:
    """The condition's ingestable metrics that are not yet stored."""
    pending: set[Metric] = set()
    for metric in _ingestable(condition, metric_ids):
        if not await writer.coval_metric_ingested(
            provider=provider, coval_run_id=coval_run_id, metric_type=metric
        ):
            pending.add(metric)
    return frozenset(pending)


async def _ingest_run(
    client: httpx.AsyncClient,
    writer: RunWriter,
    *,
    spec: AgentSpec,
    coval_run: CovalRun,
    metric_ids: Mapping[Metric, str],
    condition: DatasetMetrics = DEFAULT_CONDITION,
    pending: frozenset[Metric] | None = None,
    dataset_id: str = DATASET_ID,
    dataset_sha256: str = "",
    runner_sha: str,
    period_seconds: int,
) -> RunStatus | None:
    """Ingest one Coval run into its own run row; None = skipped, nothing written.

    Writes the metrics *condition* declares and *pending* still owes (default all
    of them). Skips (before any DB write) runs finalized with an error_status,
    runs missing the condition's required metric, and runs left with nothing to
    write. SUCCEEDED = all clips numeric, PARTIAL = some failed, FAILED = all.
    """
    pending = condition.fetched if pending is None else pending
    run_pk: int | None = None
    try:
        resp = await client.get(f"/runs/{coval_run.run_id}")
        resp.raise_for_status()
        run = cast("dict[str, Any]", resp.json()["run"])
        error_status = _error_status(run.get("error_status"))
        if error_status:
            logger.warning(
                "errored_run_skipped",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                error_status=error_status,
            )
            return None
        metrics = cast("dict[str, Any]", (run.get("results") or {}).get("metrics") or {})
        anchor = condition.required
        anchor_values = _metric_values(metrics, metric_ids.get(anchor))
        if anchor_values is None:
            # The condition says this one must be here, so its absence is a fault
            # rather than a quiet skip. Metrics the condition omits are never
            # looked up, so they never reach this branch.
            logger.warning(
                "required_metric_absent",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                dataset_id=dataset_id,
                metric=anchor.value,
            )
            return None
        if _has_duplicate_ids(anchor_values):
            # Nothing else checks the anchor's own population, and duplicate rows
            # would double-count conversations in the aggregate.
            logger.warning(
                "anchor_duplicate_ids",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                metric=anchor.value,
            )
            return None

        # Decide writability up front (pure) so a backfill with nothing to add
        # creates no run row and stays retryable.
        writable: dict[Metric, list[dict[str, Any]]] = {}
        if anchor in pending:
            writable[anchor] = anchor_values
        for metric in sorted(condition.optional):
            if metric not in pending:
                continue
            values = _metric_values(metrics, metric_ids.get(metric))
            if values is None:
                logger.info(
                    "optional_metric_absent",
                    provider=spec.provider,
                    coval_run_id=coval_run.run_id,
                    metric=metric.value,
                )
                continue
            mismatch = _population_mismatch(anchor_values, values)
            if mismatch is not None:
                # A different conversation population than the anchor would make
                # the rate incomparable, so drop this metric and keep the anchor.
                logger.warning(
                    "metric_population_mismatch",
                    provider=spec.provider,
                    coval_run_id=coval_run.run_id,
                    metric=metric.value,
                    **mismatch,
                )
                continue
            writable[metric] = values
        # Same mapper the rows are built from, so "would this write anything?"
        # cannot drift from what actually gets written.
        for metric in list(writable):
            try:
                mapped = [_VALUE_MAPPERS[metric](v.get("value")) for v in writable[metric]]
            except InvalidInstructionVerdict as exc:
                # Judge-contract violation: discard that metric for the whole run;
                # retryable on a later scan.
                logger.warning(
                    "instruction_verdict_invalid",
                    provider=spec.provider,
                    coval_run_id=coval_run.run_id,
                    metric=metric.value,
                    error=str(exc),
                )
                del writable[metric]
            else:
                if not any(m is not None for m in mapped):
                    # Every value maps to no row (all UNKNOWN), so claim nothing.
                    logger.info(
                        "metric_yields_no_rows",
                        provider=spec.provider,
                        coval_run_id=coval_run.run_id,
                        metric=metric.value,
                    )
                    del writable[metric]

        if not writable:
            # Backfill with nothing to add; leave it retryable, write no run row.
            return None

        scheduled_at = _bucket_start(coval_run.create_time or datetime.now(tz=UTC), period_seconds)
        run_row = await writer.start_run(
            runner_sha=runner_sha,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256 or _dataset_sha256(),
            scheduled_at=scheduled_at,
            persona_id=coval_run.persona_id or None,
        )
        if run_row.id is None:  # pragma: no cover -- start_run always returns an id
            raise RuntimeError("start_run returned a run with no id")
        run_pk = run_row.id

        # Grouped as built: metric_type is a plain str on Result, so filtering the
        # flat list by enum identity would silently match nothing.
        by_metric = {
            metric: _s2s_rows(
                values, metric=metric, run_pk=run_pk, coval_run_id=coval_run.run_id, spec=spec
            )
            for metric, values in writable.items()
        }
        all_rows = [row for rows in by_metric.values() for row in rows]
        rows = by_metric.get(Metric.V2V, [])
        if all_rows:
            await writer.record_results(all_rows)
        logger.info(
            "fetched_clips",
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            slot=str(scheduled_at),
            clips=len(rows),
            instruction=len(by_metric.get(Metric.INSTRUCTION_FOLLOWING, [])),
            success=sum(1 for r in rows if r.status is ResultStatus.SUCCESS),
        )
        if Metric.V2V in writable:
            # Reliability is the latency signal (other metrics never fail the run).
            failed = sum(1 for r in rows if r.status is ResultStatus.FAILED)
            if not rows or failed == len(rows):
                status = RunStatus.FAILED
            elif failed:
                status = RunStatus.PARTIAL
            else:
                status = RunStatus.SUCCEEDED
        else:
            # This condition has no latency, or it landed on an earlier scan.
            status = RunStatus.SUCCEEDED
        await writer.finish_run(run_pk, status=status)
        if status in (RunStatus.SUCCEEDED, RunStatus.PARTIAL):
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


async def _fetch_one_provider(
    client: httpx.AsyncClient,
    writer: RunWriter,
    *,
    spec: AgentSpec,
    agent_id: str,
    metric_ids: Mapping[Metric, str],
    test_set_id: str | None = None,
    noisy_persona_id: str | None = None,
    runner_sha: str,
    period_seconds: int,
    stale_grace_seconds: int,
    sampled_runs: list[SampleRun] | None = None,
) -> tuple[RunStatus, int]:
    """Scan the window and ingest every clean, not-yet-ingested run.

    Returns (provider status, runs ingested this tick). Staleness wins: when
    the newest usable run is unknown-age or older than period + grace, the
    provider is FAILED even if old runs were backfilled this tick, so a stuck
    sim, a paused schedule, or a Coval outage is always loud. Errors are
    caught here so one provider can't abort others.
    """
    statuses: list[RunStatus] = []
    candidates: dict[tuple[datetime, str], SampleRun] = {}

    def note_sample_candidate(coval_run: CovalRun) -> None:
        # Newest-first scan: keep the newest eligible run PER (BUCKET, PERSONA).
        # Multi-turn runs the same test set once per persona, so a provider yields
        # one candidate per persona per day (single-turn has no persona -> a "" key).
        # Keying on the bucket too is what lets a missed day still be published: the
        # window spans two periods, and keying on persona alone let today's run evict
        # yesterday's before the sampler ever saw it. Staggered arrivals must not
        # shrink the sample; committed only after the staleness check so a stale
        # provider's old recording never ships today.
        # The noisy caller is under embargo, so its recordings never reach the
        # public samples card.
        if noisy_persona_id and coval_run.persona_id == noisy_persona_id:
            return
        bucket_at = _bucket_start(coval_run.create_time or datetime.now(tz=UTC), period_seconds)
        key = (bucket_at, coval_run.persona_id)
        if key in candidates:
            return
        candidates[key] = SampleRun(
            provider=spec.provider,
            model=spec.model,
            coval_run_id=coval_run.run_id,
            bucket_at=bucket_at,
            persona_id=coval_run.persona_id,
            agent_id=agent_id,
        )

    try:
        runs = await recent_completed_runs(
            client, agent_id, period_seconds=period_seconds, test_set_id=test_set_id
        )
        data_seen = False
        newest_data_at: datetime | None = None
        for coval_run in runs:
            if coval_run.error_status:
                logger.warning(
                    "errored_run_skipped",
                    provider=spec.provider,
                    coval_run_id=coval_run.run_id,
                    error_status=coval_run.error_status,
                )
                continue
            dataset_id, dataset_sha256 = _dataset_identity(
                test_set_id, coval_run.persona_id, noisy_persona_id
            )
            condition = condition_for(dataset_id)
            if condition.required not in metric_ids:
                logger.warning(
                    "required_metric_unconfigured",
                    provider=spec.provider,
                    dataset_id=dataset_id,
                    metric=condition.required.value,
                )
                continue
            ingestable = _ingestable(condition, metric_ids)
            pending = await _pending_metrics(
                writer,
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                condition=condition,
                metric_ids=metric_ids,
            )
            persisted = ingestable - pending
            if persisted:
                note_sample_candidate(coval_run)
            if condition.required in persisted and not data_seen:
                data_seen, newest_data_at = True, coval_run.create_time
            if not pending:
                logger.info(
                    "run_already_ingested", provider=spec.provider, coval_run_id=coval_run.run_id
                )
                continue
            status = await _ingest_run(
                client,
                writer,
                spec=spec,
                coval_run=coval_run,
                metric_ids=metric_ids,
                condition=condition,
                pending=pending,
                dataset_id=dataset_id,
                dataset_sha256=dataset_sha256,
                runner_sha=runner_sha,
                period_seconds=period_seconds,
            )
            if status is None:
                continue
            if status is not RunStatus.FAILED:
                note_sample_candidate(coval_run)
                if not data_seen:
                    data_seen, newest_data_at = True, coval_run.create_time
            statuses.append(status)

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

        if sampled_runs is not None:
            sampled_runs.extend(candidates.values())

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


async def fetch_and_write_v2v(settings: Settings | None = None) -> dict[str, RunStatus]:
    """Ingest every provider's recent runs; return per-provider status.

    Each ingested Coval run gets its own run row slotted by its create_time,
    and ``coval_run_ingested`` makes re-scans no-ops, so ticks are idempotent
    and the fetch cadence only affects how soon data appears — the cron may
    run more often than the sims.
    """
    settings = settings or get_settings()

    metric_id = settings.coval_s2s_latency_metric_id
    if not metric_id:
        raise RuntimeError("coval_s2s_latency_metric_id is not set")
    # Instruction ingestion and the test-set filter go together: instruction
    # without the filter would pool other sims on the same agents into the S2S
    # rows. Reject a blank (misconfigured) value and require the pair; both
    # absent keeps the legacy latency-only behavior.
    raw_instr = settings.coval_s2s_instruction_metric_id
    raw_test_set = settings.coval_s2s_test_set_id
    if (raw_instr is not None and not raw_instr.strip()) or (
        raw_test_set is not None and not raw_test_set.strip()
    ):
        raise RuntimeError(
            "coval_s2s_instruction_metric_id / coval_s2s_test_set_id must not be blank"
        )
    instruction_metric_id = raw_instr or None
    test_set_id = raw_test_set or None
    if bool(instruction_metric_id) != bool(test_set_id):
        raise RuntimeError(
            "coval_s2s_instruction_metric_id and coval_s2s_test_set_id must be set together"
        )
    # The noisy persona only separates conditions within a test set, so without
    # one it would silently never take effect.
    raw_noisy = settings.coval_s2s_noisy_persona_id
    if raw_noisy is not None and not raw_noisy.strip():
        raise RuntimeError("coval_s2s_noisy_persona_id must not be blank")
    noisy_persona_id = raw_noisy or None
    if noisy_persona_id and not test_set_id:
        raise RuntimeError("coval_s2s_noisy_persona_id requires coval_s2s_test_set_id")
    raw_interruption = settings.coval_s2s_interruption_metric_id
    if raw_interruption is not None and not raw_interruption.strip():
        raise RuntimeError("coval_s2s_interruption_metric_id must not be blank")
    # Only configured metrics are ever asked for, so an unset id simply means that
    # metric is not ingested yet.
    metric_ids: dict[Metric, str] = {Metric.V2V: metric_id}
    if instruction_metric_id:
        metric_ids[Metric.INSTRUCTION_FOLLOWING] = instruction_metric_id
    if raw_interruption:
        metric_ids[Metric.INTERRUPTION_RATE] = raw_interruption

    async with _client(settings) as client, lifespan_pool(settings) as pool:
        writer = RunWriter(pool)
        statuses: dict[str, RunStatus] = {}
        total_ingested = 0
        sampled_runs: list[SampleRun] = []
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
                metric_ids=metric_ids,
                test_set_id=test_set_id,
                noisy_persona_id=noisy_persona_id,
                runner_sha=settings.runner_sha,
                period_seconds=settings.s2s_fetch_period_seconds,
                stale_grace_seconds=settings.s2s_stale_grace_seconds,
                sampled_runs=sampled_runs,
            )
            total_ingested += ingested

        if total_ingested:
            try:
                await writer.refresh_stats_matviews()
            except Exception:
                logger.warning("refresh_stats_matviews_failed", exc_info=True)

        if settings.s2s_samples_bucket and sampled_runs and test_set_id:
            expected = {
                (spec.provider, spec.model)
                for spec in AGENTS
                if getattr(settings, spec.agent_id_attr)
            }
            missing = expected - {r.key for r in sampled_runs}
            if missing:
                # Error level on purpose: this is the alert that a model is
                # absent from the window, so no day in it can publish a sample.
                logger.error("samples_provider_missing", missing=model_labels(missing))
            await publish_tick_sample(
                client,
                bucket_name=settings.s2s_samples_bucket,
                test_set_id=test_set_id,
                runs=sampled_runs,
                rng=random.Random(),  # noqa: S311
                expected_models=expected,
            )
        logger.info(
            "s2s_fetch_done",
            statuses={p: str(s) for p, s in statuses.items()},
            ingested=total_ingested,
        )
        return statuses


@click.command(name="fetch-s2s")
def fetch_s2s() -> None:
    """Fetch S2S latency from Coval and write per-clip rows (scheduled Cloud Run Job)."""
    from coval_bench.logging import configure_logging, log_run_failed, log_run_partial

    settings = get_settings()
    configure_logging(level=settings.log_level)
    # A setup crash fails the whole job.
    try:
        statuses = asyncio.run(fetch_and_write_v2v(settings))
    except Exception as exc:
        log_run_failed(str(exc), exc)
        raise

    # Healthy providers' rows are already committed, so a PARTIAL run alerts but
    # still exits 0 (no Cloud Run retry). Only a total loss fails the job.
    failed = [p for p, s in statuses.items() if s is RunStatus.FAILED]
    if not statuses or all(s is RunStatus.FAILED for s in statuses.values()):
        if statuses:
            log_run_failed(f"s2s fetch failed for all providers: {', '.join(failed)}")
        else:
            log_run_failed("s2s fetch ran no providers (none configured)")
        raise click.ClickException("s2s fetch failed for all providers")
    if failed:
        log_run_partial(f"s2s fetch has no fresh data from: {', '.join(failed)}")


if __name__ == "__main__":
    fetch_s2s()
