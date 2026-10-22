# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingest one completed Coval run into its own run row."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import structlog
from google.cloud import storage

from coval_bench.db.models import (
    CAPTURE_PENDING_ERROR,
    MetricExecutor,
    Result,
    ResultStatus,
    Run,
    RunStatus,
)
from coval_bench.db.writer import RunWriter
from coval_bench.registries import Metric
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.runner.capture import (
    ImportRunClaim,
    ImportRunIdentity,
    RunSeal,
    read_import_run_claim,
    read_run_state,
    upload_import_run_claim,
)
from coval_bench.s2s.conditions import (
    DATASET_ID,
    DEFAULT_CONDITION,
    DatasetMetrics,
)
from coval_bench.s2s.coval_runs import CovalRun, bucket_start
from coval_bench.s2s.identity import fingerprint_for, packaged_dataset_sha256
from coval_bench.s2s.rows import (
    LOCAL_SOURCES,
    VALUE_MAPPERS,
    InvalidInstructionVerdict,
    failed_conversation_rows,
    has_duplicate_ids,
    ingestable_metrics,
    local_rows,
    metric_values,
    population_mismatch,
    s2s_rows,
)
from coval_bench.s2s.specs import AgentSpec

logger = structlog.get_logger("coval_bench.s2s.ingest")


async def pending_metrics(
    writer: RunWriter,
    *,
    benchmark: Benchmark,
    provider: str,
    coval_run_id: str,
    condition: DatasetMetrics,
    metric_ids: Mapping[Metric, str],
) -> frozenset[Metric]:
    """The condition's ingestable metrics that are not yet stored."""
    pending: set[Metric] = set()
    for metric in ingestable_metrics(condition, metric_ids):
        if not await writer.coval_metric_ingested(
            benchmark=benchmark,
            provider=provider,
            coval_run_id=coval_run_id,
            metric_type=metric,
        ):
            pending.add(metric)
    return frozenset(pending)


def import_identity(
    *,
    spec: AgentSpec,
    coval_run: CovalRun,
    dataset_id: str,
    dataset_sha256: str,
    workspace_id: str | None,
) -> ImportRunIdentity:
    return ImportRunIdentity(
        external_run_id=coval_run.run_id,
        workspace_id=workspace_id or "",
        benchmark=str(spec.benchmark),
        provider=spec.provider,
        model=spec.model,
        dataset_id=dataset_id,
        dataset_sha256=fingerprint_for(dataset_sha256),
        persona_id=coval_run.persona_id,
    )


async def resolve_import_generation(
    client: storage.Client,
    bucket: str,
    identity: ImportRunIdentity,
) -> tuple[ImportRunClaim | None, int | None]:
    """Return an unsealed generation to resume, or the next generation number.

    A sealed generation without a finalization receipt is owned by the recovery
    command.  Starting another generation while it is unresolved could publish
    an incomplete predecessor.
    """
    generation = 0
    while True:
        claim = await asyncio.to_thread(read_import_run_claim, client, bucket, identity, generation)
        if claim is None:
            return None, generation
        finalized = await asyncio.to_thread(
            read_run_state, client, bucket, claim.run_id, "finalized"
        )
        if finalized is not None:
            generation += 1
            continue
        seal = await asyncio.to_thread(read_run_state, client, bucket, claim.run_id, "seal")
        if seal is not None:
            return None, None
        return claim, generation


def _writable_metrics(
    *,
    spec: AgentSpec,
    coval_run: CovalRun,
    metrics: dict[str, Any],
    anchor: Metric,
    anchor_values: list[dict[str, Any]],
    condition: DatasetMetrics,
    pending: frozenset[Metric],
    metric_ids: Mapping[Metric, str],
) -> dict[Metric, list[dict[str, Any]]]:
    """The fetched metrics worth writing, trimmed to the anchor's conversations.

    Decided up front and purely, so a backfill with nothing to add creates no run
    row and stays retryable.
    """
    writable: dict[Metric, list[dict[str, Any]]] = {}
    if anchor in pending:
        writable[anchor] = anchor_values
    for metric in sorted(condition.optional):
        if metric not in pending:
            continue
        values = metric_values(metrics, metric_ids.get(metric))
        if values is None:
            logger.info(
                "optional_metric_absent",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                metric=metric.value,
            )
            continue
        # An id-less value can't be reconciled against the anchor or deduped, and
        # its positional key is not stable across metrics, so it goes first.
        values = [v for v in values if v.get("simulation_output_id")]
        mismatch = population_mismatch(anchor_values, values)
        if mismatch is not None:
            logger.warning(
                "metric_population_mismatch",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                metric=metric.value,
                **mismatch,
            )
            # Trim to the overlap rather than drop the metric: one unmeasured call
            # should cost that call, not the other forty-nine. Duplicates are
            # corrupt, though: one conversation would carry twice its weight.
            if mismatch["duplicate_ids"]:
                continue
            anchor_ids = {sid for v in anchor_values if (sid := v.get("simulation_output_id"))}
            values = [v for v in values if v.get("simulation_output_id") in anchor_ids]
            if not values:
                continue
        writable[metric] = values
    # Same mapper the rows are built from, so "would this write anything?" can't
    # drift from what gets written.
    for metric in list(writable):
        try:
            mapped = [VALUE_MAPPERS[metric](v.get("value")) for v in writable[metric]]
        except InvalidInstructionVerdict as exc:
            # Judge-contract violation: drop the metric for this run, retry later.
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
                # All UNKNOWN: nothing to claim.
                logger.info(
                    "metric_yields_no_rows",
                    provider=spec.provider,
                    coval_run_id=coval_run.run_id,
                    metric=metric.value,
                )
                del writable[metric]
    return writable


async def _local_metrics(
    writer: RunWriter,
    *,
    spec: AgentSpec,
    coval_run: CovalRun,
    condition: DatasetMetrics,
    pending: frozenset[Metric],
    sim_ids: list[str],
) -> dict[Metric, Mapping[str, float]]:
    """Metrics measured on our side, looked up by the anchor's conversation ids.

    A metric with no values yet stays pending for a later scan. Once any
    conversation has one the run counts as ingested, and the rest is a permanent
    coverage gap, as an UNKNOWN verdict is for instruction.
    """
    local: dict[Metric, Mapping[str, float]] = {}
    for metric in sorted(condition.local & pending):
        measured = await LOCAL_SOURCES[metric](writer, sim_ids)
        if not measured:
            continue
        if len(measured) < len(sim_ids):
            logger.warning(
                "local_metric_coverage_gap",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                metric=metric.value,
                conversations=len(sim_ids) - len(measured),
            )
        local[metric] = measured
    return local


async def _claim_run(
    writer: RunWriter,
    *,
    artifact_client: storage.Client,
    artifact_bucket: str,
    import_identity: ImportRunIdentity,
    import_generation: int,
    import_claim: ImportRunClaim | None,
    scheduled_at: datetime,
    metric_types: list[str],
) -> ImportRunClaim:
    """Reserve a run id and record the claim, or reuse the claim being resumed."""
    if import_claim is None:
        started_at = datetime.now(UTC)
        return await asyncio.to_thread(
            upload_import_run_claim,
            artifact_client,
            artifact_bucket,
            ImportRunClaim(
                identity=import_identity,
                generation=import_generation,
                run_id=await writer.reserve_run_id(),
                started_at=started_at,
                scheduled_at=scheduled_at,
                captured_at=started_at,
                metric_types=metric_types,
            ),
        )
    if import_claim.identity != import_identity or import_claim.generation != import_generation:
        raise ValueError("external import claim does not match the requested generation")
    return import_claim


def _rows_by_metric(
    *,
    spec: AgentSpec,
    run_pk: int,
    coval_run: CovalRun,
    run: dict[str, Any],
    writable: Mapping[Metric, list[dict[str, Any]]],
    local: Mapping[Metric, Mapping[str, float]],
) -> dict[Metric, list[Result]]:
    """Rows grouped as built: metric_type is a plain str on Result, so filtering the
    flat list by enum identity would silently match nothing."""
    by_metric = {
        metric: s2s_rows(
            values, metric=metric, run_pk=run_pk, coval_run_id=coval_run.run_id, spec=spec
        )
        for metric, values in writable.items()
    }
    for metric, local_values in local.items():
        by_metric[metric] = local_rows(
            local_values, metric=metric, run_pk=run_pk, coval_run_id=coval_run.run_id, spec=spec
        )
    # Optional V2V (the workflow sets) gets the same FAILED rows as anchored V2V,
    # or its latency would cover only the calls a model survived.
    if Metric.V2V in writable:
        output_ids = [
            s
            for s in cast("list[Any]", (run.get("results") or {}).get("output_ids") or [])
            if isinstance(s, str)
        ]
        by_metric[Metric.V2V].extend(
            failed_conversation_rows(
                output_ids,
                writable[Metric.V2V],
                run_pk=run_pk,
                coval_run_id=coval_run.run_id,
                spec=spec,
            )
        )
    return by_metric


async def _capture_rows(
    writer: RunWriter,
    *,
    artifact_client: storage.Client,
    artifact_bucket: str,
    spec: AgentSpec,
    coval_run: CovalRun,
    run_pk: int,
    dataset_id: str,
    dataset_fingerprint: str,
    scheduled_at: datetime,
    captured_at: datetime,
    all_rows: list[Result],
) -> tuple[list[Any], list[str]]:
    """Upload the run manifest, then persist one capture envelope per conversation.

    Returns the capture outcomes and the expected capture ids for the seal.
    """
    from coval_bench.runner.capture import (
        RunManifest,
        build_capture_identity,
        identity_digest,
        upload_run_state,
    )
    from coval_bench.runner.persistence import persist_capture, prepare_capture_envelope

    grouped: dict[str, list[Result]] = {}
    for row in all_rows:
        grouped.setdefault(cast("str", row.audio_filename), []).append(row)
    expected_capture_ids = [
        identity_digest(
            build_capture_identity(
                run_id=run_pk,
                benchmark=spec.benchmark.value,
                dataset_id=dataset_id,
                sample_id=sample_id,
                provider=spec.provider,
                model=spec.model,
                capture_id=f"{coval_run.run_id}:{sample_id}",
            )
        )
        for sample_id in sorted(grouped)
    ]
    await asyncio.to_thread(
        upload_run_state,
        artifact_client,
        artifact_bucket,
        run_pk,
        "manifest",
        RunManifest(
            run_id=run_pk,
            scheduled_at=scheduled_at,
            benchmark_kind=spec.benchmark.value.lower(),
            source="coval-api",
            datasets={dataset_id: dataset_fingerprint},
            expected_capture_ids=expected_capture_ids,
        ),
    )
    capture_outcomes: list[Any] = []
    for index, (sample_id, sample_rows) in enumerate(sorted(grouped.items())):
        provider_error = None
        if not any(row.status is ResultStatus.SUCCESS for row in sample_rows):
            provider_error = next(
                (row.error for row in sample_rows if row.error),
                "Coval conversation produced no successful metric value",
            )
        envelope = prepare_capture_envelope(
            run_id=run_pk,
            dataset_id=dataset_id,
            dataset_sha256=dataset_fingerprint,
            sample_id=sample_id,
            entry=spec,
            benchmark=spec.benchmark,
            results=sample_rows,
            provider_error=provider_error,
            captured_at=captured_at + timedelta(microseconds=index),
            executor=MetricExecutor.COVAL_API,
            capture_id=f"{coval_run.run_id}:{sample_id}",
            provider_extras={"coval_run_id": coval_run.run_id},
        )
        capture_outcomes.append(
            await persist_capture(
                writer=writer,
                storage_client=artifact_client,
                bucket=artifact_bucket,
                envelope=envelope,
            )
        )
    return capture_outcomes, expected_capture_ids


def _latency_status(writable: Mapping[Metric, Any], rows: list[Result]) -> RunStatus:
    """Reliability is the latency signal; other metrics never fail the run."""
    if Metric.V2V not in writable:
        # This condition has no latency, or it landed on an earlier scan.
        return RunStatus.SUCCEEDED
    failed = sum(1 for r in rows if r.status is ResultStatus.FAILED)
    if not rows or failed == len(rows):
        return RunStatus.FAILED
    if failed:
        return RunStatus.PARTIAL
    return RunStatus.SUCCEEDED


async def _seal_import(
    writer: RunWriter,
    *,
    artifact_client: storage.Client,
    artifact_bucket: str,
    spec: AgentSpec,
    run_pk: int,
    status: RunStatus,
    claimed_run: Run,
    capture_outcomes: list[Any],
    expected_capture_ids: list[str],
) -> RunStatus:
    """Seal and finish the run row; a resumed run keeps the status it already stored."""
    from coval_bench.runner.persistence import capture_pending, seal_and_finish_run

    incomplete = capture_pending(capture_outcomes, expected_capture_ids)
    held = incomplete and status is not RunStatus.FAILED
    if claimed_run.finished_at is not None:
        stored_status, finish_error = claimed_run.status, claimed_run.error
        finished_at = claimed_run.finished_at
    else:
        stored_status = RunStatus.PARTIAL if held else status
        finish_error = CAPTURE_PENDING_ERROR if held else None
        finished_at = datetime.now(UTC)
    status = await seal_and_finish_run(
        writer=writer,
        storage_client=artifact_client,
        bucket=artifact_bucket,
        seal=RunSeal(
            run_id=run_pk,
            intended_status=str(status),
            stored_status=str(stored_status),
            finished_at=finished_at,
            error=finish_error,
            expected_capture_ids=expected_capture_ids,
        ),
        capture_pending=incomplete,
    )
    if status in (RunStatus.SUCCEEDED, RunStatus.PARTIAL):
        try:
            await writer.rebuild_run_rollup(run_pk)
        except Exception:
            logger.warning(
                "normalized_bucket_refresh_failed", provider=spec.provider, exc_info=True
            )
    return status


async def ingest_run(
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
    period_seconds: int,
    artifact_client: storage.Client,
    artifact_bucket: str,
    workspace_id: str | None = None,
    import_identity: ImportRunIdentity,
    import_generation: int,
    import_claim: ImportRunClaim | None = None,
) -> RunStatus | None:
    """Ingest one Coval run into its own run row; None means skipped, nothing written.

    Writes what *condition* declares and *pending* still owes (default: all of it).
    Coval's error_status is ignored on purpose: one failed conversation stamps the
    whole run EXECUTION_FAILURE, and gating on it threw away healthy clips over a
    single short call. A conversation with no anchor value becomes a FAILED row
    instead. SUCCEEDED = all clips numeric, PARTIAL = some failed, FAILED = all.
    """
    if pending is None:
        pending = condition.fetched | (condition.local & frozenset(LOCAL_SOURCES))
    if import_claim is not None:
        pending = frozenset(Metric(metric) for metric in import_claim.metric_types)
    run_pk: int | None = None
    headers = {"X-Coval-Workspace-Id": workspace_id} if workspace_id else None
    try:
        resp = await client.get(f"/runs/{coval_run.run_id}", headers=headers)
        resp.raise_for_status()
        run = cast("dict[str, Any]", resp.json()["run"])
        metrics = cast("dict[str, Any]", (run.get("results") or {}).get("metrics") or {})
        anchor = condition.required
        anchor_values = metric_values(metrics, metric_ids.get(anchor))
        if anchor_values is None:
            # The condition says this one must be here, so its absence is a fault.
            logger.warning(
                "required_metric_absent",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                dataset_id=dataset_id,
                metric=anchor.value,
            )
            return None
        if has_duplicate_ids(anchor_values):
            # Nothing else checks the anchor's own population, and duplicate rows
            # would double-count conversations in the aggregate.
            logger.warning(
                "anchor_duplicate_ids",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                metric=anchor.value,
            )
            return None
        writable = _writable_metrics(
            spec=spec,
            coval_run=coval_run,
            metrics=metrics,
            anchor=anchor,
            anchor_values=anchor_values,
            condition=condition,
            pending=pending,
            metric_ids=metric_ids,
        )
        local = await _local_metrics(
            writer,
            spec=spec,
            coval_run=coval_run,
            condition=condition,
            pending=pending,
            sim_ids=[sid for v in anchor_values if (sid := v.get("simulation_output_id"))],
        )
        if not writable and not local:
            # Backfill with nothing to add; write no run row so it stays retryable.
            return None

        run_dataset_sha256 = dataset_sha256 or packaged_dataset_sha256()
        claim = await _claim_run(
            writer,
            artifact_client=artifact_client,
            artifact_bucket=artifact_bucket,
            import_identity=import_identity,
            import_generation=import_generation,
            import_claim=import_claim,
            scheduled_at=bucket_start(
                coval_run.create_time or datetime.now(tz=UTC), period_seconds
            ),
            metric_types=sorted(str(metric) for metric in set(writable) | set(local)),
        )
        run_pk = claim.run_id
        claimed_run = await writer.ensure_capture_run(
            run_pk,
            started_at=claim.started_at,
            dataset_id=dataset_id,
            dataset_sha256=run_dataset_sha256,
            scheduled_at=claim.scheduled_at,
            persona_id=coval_run.persona_id or None,
        )
        by_metric = _rows_by_metric(
            spec=spec, run_pk=run_pk, coval_run=coval_run, run=run, writable=writable, local=local
        )
        all_rows = [row for rows in by_metric.values() for row in rows]
        rows = by_metric.get(Metric.V2V, [])
        capture_outcomes: list[Any] = []
        expected_capture_ids: list[str] = []
        if all_rows:
            capture_outcomes, expected_capture_ids = await _capture_rows(
                writer,
                artifact_client=artifact_client,
                artifact_bucket=artifact_bucket,
                spec=spec,
                coval_run=coval_run,
                run_pk=run_pk,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint_for(run_dataset_sha256),
                scheduled_at=claim.scheduled_at,
                captured_at=claim.captured_at,
                all_rows=all_rows,
            )
        logger.info(
            "fetched_clips",
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            slot=str(claim.scheduled_at),
            clips=len(rows),
            instruction=len(by_metric.get(Metric.INSTRUCTION_FOLLOWING, [])),
            workflow=len(by_metric.get(Metric.WORKFLOW_ADHERENCE, [])),
            ttft=len(by_metric.get(Metric.TTFT, [])),
            success=sum(1 for r in rows if r.status is ResultStatus.SUCCESS),
        )
        return await _seal_import(
            writer,
            artifact_client=artifact_client,
            artifact_bucket=artifact_bucket,
            spec=spec,
            run_pk=run_pk,
            status=_latency_status(writable, rows),
            claimed_run=claimed_run,
            capture_outcomes=capture_outcomes,
            expected_capture_ids=expected_capture_ids,
        )
    except Exception as exc:
        if run_pk is not None:
            logger.warning(
                "normalized_import_capture_pending",
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                run_id=run_pk,
                error=str(exc),
            )
            return RunStatus.PARTIAL
        logger.warning(
            "run_ingest_failed",
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            error=str(exc),
        )
        return RunStatus.FAILED
