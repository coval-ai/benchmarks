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
from coval_bench.registries import Metric
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.samples import SampleRun, copy_tick_samples

logger = structlog.get_logger("coval_bench.s2s.fetch_v2v")

# The dataset the S2S sims run against (matches the packaged manifest name).
DATASET_ID = "s2s-v1"
# Multi-turn runs have no local manifest -- they're keyed to the Coval test set --
# so they use a distinct dataset id and never pool with single-turn s2s-v1.
DATASET_ID_MULTITURN = "s2s-multiturn-v1"

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


# Agent ids resolved from Settings; model strings are display labels.
AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(agent_id_attr="coval_s2s_openai_agent_id", provider="openai", model="gpt-realtime"),
    AgentSpec(agent_id_attr="coval_s2s_gemini_agent_id", provider="google", model="gemini-live"),
    AgentSpec(agent_id_attr="coval_s2s_xai_agent_id", provider="xai", model="grok-realtime"),
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


def _dataset_identity(test_set_id: str | None) -> tuple[str, str]:
    """Dataset id + provenance for the fetched rows.

    Multi-turn runs are keyed to their Coval test set, not the single-turn SLURP
    manifest, so they get their own dataset id (never pooling with s2s-v1) and
    record the test-set id as provenance. Without a test set (legacy latency-only
    mode) the rows stay under the packaged s2s-v1 manifest.
    """
    if test_set_id:
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
        )
        if run.create_time is not None and (now - run.create_time).total_seconds() > window_seconds:
            continue
        runs.append(run)
    return runs


def _result_rows(
    values: list[dict[str, Any]],
    *,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """Map a Coval run's per-clip values to Result rows.

    Values are seconds; convert to ms. A clip with no numeric value becomes a
    FAILED row, so reliability = success/total.
    """
    rows: list[Result] = []
    for i, v in enumerate(values):
        raw = v.get("value")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            metric_value: float | None = round(float(raw) * 1000, 1)
            status = ResultStatus.SUCCESS
        else:
            metric_value = None
            status = ResultStatus.FAILED
        sim_id = v.get("simulation_output_id")
        clip_key = f"{coval_run_id}/{sim_id}" if sim_id else f"{coval_run_id}/{i}"
        rows.append(
            Result(
                run_id=run_pk,
                provider=spec.provider,
                model=spec.model,
                benchmark=Benchmark.S2S,
                metric_type=Metric.V2V,
                metric_units="milliseconds",
                metric_value=metric_value,
                audio_filename=clip_key,
                status=status,
            )
        )
    return rows


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


def _instruction_id_mismatch(
    latency_values: list[dict[str, Any]], instruction_values: list[dict[str, Any]]
) -> dict[str, object] | None:
    """None if instruction's sim-id set matches latency exactly; else the diff.

    Equal counts can still be different conversations, so compare the id sets and
    reject duplicates. A mismatch means the pass rate would not be over the same
    population as latency, so instruction is skipped for the run.
    """
    lat_ids = [v.get("simulation_output_id") for v in latency_values]
    ins_ids = [v.get("simulation_output_id") for v in instruction_values]
    lat_set, ins_set = set(lat_ids), set(ins_ids)
    duplicate = len(ins_ids) != len(ins_set)
    missing = sorted(str(x) for x in lat_set - ins_set)
    extra = sorted(str(x) for x in ins_set - lat_set)
    if not duplicate and not missing and not extra:
        return None
    return {
        "missing_instruction_ids": missing,
        "extra_instruction_ids": extra,
        "duplicate_instruction_ids": duplicate,
    }


def _instruction_rows(
    values: list[dict[str, Any]],
    *,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """Map a run's per-conversation verdicts to instruction Result rows.

    YES -> 100.0, NO -> 0.0 (both SUCCESS, both in the pool); UNKNOWN produces no
    row (excluded), so the aggregate mean is YES / (YES + NO). Raises
    InvalidInstructionVerdict on any value outside the contract.
    """
    rows: list[Result] = []
    for i, v in enumerate(values):
        verdict = _instruction_verdict(v.get("value"))
        if verdict is None:  # UNKNOWN: excluded from the pool, no row written
            continue
        sim_id = v.get("simulation_output_id")
        clip_key = f"{coval_run_id}/{sim_id}" if sim_id else f"{coval_run_id}/{i}"
        rows.append(
            Result(
                run_id=run_pk,
                provider=spec.provider,
                model=spec.model,
                benchmark=Benchmark.S2S,
                metric_type=Metric.INSTRUCTION_FOLLOWING,
                metric_units="percent",
                metric_value=100.0 if verdict else 0.0,
                audio_filename=clip_key,
                status=ResultStatus.SUCCESS,
            )
        )
    return rows


async def _ingest_run(
    client: httpx.AsyncClient,
    writer: RunWriter,
    *,
    spec: AgentSpec,
    coval_run: CovalRun,
    metric_id: str,
    instruction_metric_id: str | None = None,
    dataset_id: str = DATASET_ID,
    dataset_sha256: str = "",
    want_latency: bool = True,
    want_instruction: bool = True,
    runner_sha: str,
    period_seconds: int,
) -> RunStatus | None:
    """Ingest one Coval run into its own run row; None = skipped, nothing written.

    Skips (before any DB write) runs finalized with an error_status — those
    failed upstream of the provider and must not dent its reliability — and
    runs missing the metric. Otherwise SUCCEEDED = all clips numeric, PARTIAL =
    some failed, FAILED = all failed.
    """
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
        metric = metrics.get(metric_id)
        if metric is None:
            logger.warning(
                "metric_absent",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                metric_id=metric_id,
            )
            return None
        values = cast("list[dict[str, Any]]", metric.get("values", []))

        # Decide instruction writability up front (pure) so a backfill with
        # nothing to add creates no run row and stays retryable.
        instr_values: list[dict[str, Any]] = []
        write_instruction = False
        if want_instruction and instruction_metric_id:
            instr = metrics.get(instruction_metric_id)
            if instr is None:
                # Latency still lands; instruction stays retryable on a later scan.
                logger.warning(
                    "instruction_metric_absent",
                    provider=spec.provider,
                    coval_run_id=coval_run.run_id,
                    metric_id=instruction_metric_id,
                )
            else:
                instr_values = cast("list[dict[str, Any]]", instr.get("values", []))
                mismatch = _instruction_id_mismatch(values, instr_values)
                if mismatch is not None:
                    # Different conversation population than latency -> skip
                    # instruction (keep latency) so the rate stays comparable.
                    logger.warning(
                        "instruction_id_mismatch",
                        provider=spec.provider,
                        coval_run_id=coval_run.run_id,
                        **mismatch,
                    )
                else:
                    try:
                        for v in instr_values:
                            _instruction_verdict(v.get("value"))
                        write_instruction = True
                    except InvalidInstructionVerdict as exc:
                        # Judge-contract violation: discard instruction for the
                        # whole run (keep latency); retryable on a later scan.
                        logger.warning(
                            "instruction_verdict_invalid",
                            provider=spec.provider,
                            coval_run_id=coval_run.run_id,
                            error=str(exc),
                        )

        if not want_latency and not write_instruction:
            # Backfill with nothing to add; leave it retryable, write no run row.
            return None

        scheduled_at = _bucket_start(coval_run.create_time or datetime.now(tz=UTC), period_seconds)
        run_row = await writer.start_run(
            runner_sha=runner_sha,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256 or _dataset_sha256(),
            scheduled_at=scheduled_at,
        )
        if run_row.id is None:  # pragma: no cover -- start_run always returns an id
            raise RuntimeError("start_run returned a run with no id")
        run_pk = run_row.id

        rows = (
            _result_rows(values, run_pk=run_pk, coval_run_id=coval_run.run_id, spec=spec)
            if want_latency
            else []
        )
        instruction_rows = (
            _instruction_rows(instr_values, run_pk=run_pk, coval_run_id=coval_run.run_id, spec=spec)
            if write_instruction
            else []
        )
        all_rows = rows + instruction_rows
        if all_rows:
            await writer.record_results(all_rows)
        logger.info(
            "fetched_clips",
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            slot=str(scheduled_at),
            clips=len(rows),
            instruction=len(instruction_rows),
            success=sum(1 for r in rows if r.status is ResultStatus.SUCCESS),
        )
        if want_latency:
            # Reliability is the latency signal (instruction lag never fails the run).
            failed = sum(1 for r in rows if r.status is ResultStatus.FAILED)
            if not rows or failed == len(rows):
                status = RunStatus.FAILED
            elif failed:
                status = RunStatus.PARTIAL
            else:
                status = RunStatus.SUCCEEDED
        else:
            # Instruction-only backfill onto an already-ingested run.
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
    metric_id: str,
    instruction_metric_id: str | None = None,
    test_set_id: str | None = None,
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
    candidate: SampleRun | None = None

    def note_sample_candidate(coval_run: CovalRun) -> None:
        # Newest-first scan: the first eligible run per provider is its newest,
        # whether it was ingested this tick or on an earlier one — staggered
        # arrivals must not shrink the sample to one provider. Held locally and
        # committed only after the staleness check, so a stale provider's old
        # recording never ships under today's tick.
        nonlocal candidate
        if candidate is not None:
            return
        candidate = SampleRun(
            provider=spec.provider,
            model=spec.model,
            coval_run_id=coval_run.run_id,
            bucket_at=_bucket_start(coval_run.create_time or datetime.now(tz=UTC), period_seconds),
        )

    try:
        runs = await recent_completed_runs(
            client, agent_id, period_seconds=period_seconds, test_set_id=test_set_id
        )
        dataset_id, dataset_sha256 = _dataset_identity(test_set_id)

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
            latency_done = await writer.coval_metric_ingested(
                provider=spec.provider, coval_run_id=coval_run.run_id, metric_type=Metric.V2V
            )
            instruction_done = instruction_metric_id is None or await writer.coval_metric_ingested(
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                metric_type=Metric.INSTRUCTION_FOLLOWING,
            )
            # A run whose latency already landed is fresh data + an eligible
            # sample even if we still owe it an instruction backfill.
            if latency_done:
                note_sample_candidate(coval_run)
                if not data_seen:
                    data_seen, newest_data_at = True, coval_run.create_time
            if latency_done and instruction_done:
                logger.info(
                    "run_already_ingested", provider=spec.provider, coval_run_id=coval_run.run_id
                )
                continue
            status = await _ingest_run(
                client,
                writer,
                spec=spec,
                coval_run=coval_run,
                metric_id=metric_id,
                instruction_metric_id=instruction_metric_id,
                dataset_id=dataset_id,
                dataset_sha256=dataset_sha256,
                want_latency=not latency_done,
                want_instruction=instruction_metric_id is not None and not instruction_done,
                runner_sha=runner_sha,
                period_seconds=period_seconds,
            )
            if status is None:
                continue
            if status is not RunStatus.FAILED:
                note_sample_candidate(coval_run)
            statuses.append(status)
            if not data_seen:
                data_seen, newest_data_at = True, coval_run.create_time

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

        if sampled_runs is not None and candidate is not None:
            sampled_runs.append(candidate)

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
            statuses[spec.provider], ingested = await _fetch_one_provider(
                client,
                writer,
                spec=spec,
                agent_id=agent_id,
                metric_id=metric_id,
                instruction_metric_id=instruction_metric_id,
                test_set_id=test_set_id,
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

        if settings.s2s_samples_bucket and sampled_runs:
            expected = {spec.provider for spec in AGENTS if getattr(settings, spec.agent_id_attr)}
            missing = expected - {r.provider for r in sampled_runs}
            if missing:
                # Error level on purpose: this is the alert that a provider is
                # absent from the day's sample; the tick still publishes the rest.
                logger.error("samples_provider_missing", missing=sorted(missing))
            await copy_tick_samples(
                client,
                bucket_name=settings.s2s_samples_bucket,
                runs=sampled_runs,
                rng=random.Random(),  # noqa: S311
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
    from coval_bench.logging import configure_logging, log_run_failed

    settings = get_settings()
    configure_logging(level=settings.log_level)
    # A setup crash fails the whole job.
    try:
        statuses = asyncio.run(fetch_and_write_v2v(settings))
    except Exception as exc:
        log_run_failed(str(exc), exc)
        raise

    # Alert if any provider has no fresh data. The succeeding providers' rows
    # are already committed, so a partial run still exits 0 — only a total
    # loss fails the job (non-zero exit).
    failed = [p for p, s in statuses.items() if s is RunStatus.FAILED]
    if not statuses:
        log_run_failed("s2s fetch ran no providers (none configured)")
    elif failed:
        log_run_failed(f"s2s fetch has no fresh data from: {', '.join(failed)}")
    if not statuses or all(s is RunStatus.FAILED for s in statuses.values()):
        raise click.ClickException("s2s fetch failed for all providers")


if __name__ == "__main__":
    fetch_s2s()
