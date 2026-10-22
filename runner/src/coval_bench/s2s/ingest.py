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
    """Ingest one Coval run into its own run row; None = skipped, nothing written.

    Writes the metrics *condition* declares and *pending* still owes (default all
    of them). Skips (before any DB write) runs missing the condition's required
    metric and runs left with nothing to write. Coval's error_status is ignored
    on purpose: one failed conversation stamps the whole run EXECUTION_FAILURE,
    and gating on it threw away the healthy clips (and starved the sampler) over
    a single short call. A conversation with no anchor value becomes a FAILED
    row instead. SUCCEEDED = all clips numeric, PARTIAL = some failed, FAILED = all.
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

        # Decide writability up front (pure) so a backfill with nothing to add
        # creates no run row and stays retryable.
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
            # An id-less value cannot be reconciled against the anchor or deduped,
            # and its positional key is not stable across metrics, so it is
            # rejected before the populations are compared.
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
                # Trim to the conversations both cover rather than discarding the
                # metric for the whole run: one unmeasured call should cost that
                # call, not the other forty-nine. Row-level parity with the anchor
                # was never exact anyway — an UNKNOWN verdict writes no row — so a
                # coverage gap is thinner data, not corrupt data. Duplicates are
                # corrupt: one conversation would carry twice its weight.
                if mismatch["duplicate_ids"]:
                    continue
                anchor_ids = {sid for v in anchor_values if (sid := v.get("simulation_output_id"))}
                values = [v for v in values if v.get("simulation_output_id") in anchor_ids]
                if not values:
                    continue
            writable[metric] = values
        # Same mapper the rows are built from, so "would this write anything?"
        # cannot drift from what actually gets written.
        for metric in list(writable):
            try:
                mapped = [VALUE_MAPPERS[metric](v.get("value")) for v in writable[metric]]
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

        # Local metrics are looked up for the anchor's conversations. A metric with
        # no values at all stays pending for a later scan; once any conversation
        # has a value the run counts as ingested, and the ones still missing are a
        # permanent coverage gap, as an UNKNOWN verdict is for instruction.
        local: dict[Metric, Mapping[str, float]] = {}
        sim_ids = [sid for v in anchor_values if (sid := v.get("simulation_output_id"))]
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

        if not writable and not local:
            # Backfill with nothing to add; leave it retryable, write no run row.
            return None

        run_dataset_sha256 = dataset_sha256 or packaged_dataset_sha256()
        scheduled_at = bucket_start(coval_run.create_time or datetime.now(tz=UTC), period_seconds)
        claim = import_claim
        if claim is None:
            started_at = datetime.now(UTC)
            reserved_run_id = await writer.reserve_run_id()
            claim = await asyncio.to_thread(
                upload_import_run_claim,
                artifact_client,
                artifact_bucket,
                ImportRunClaim(
                    identity=import_identity,
                    generation=import_generation,
                    run_id=reserved_run_id,
                    started_at=started_at,
                    scheduled_at=scheduled_at,
                    captured_at=started_at,
                    metric_types=sorted(str(metric) for metric in set(writable) | set(local)),
                ),
            )
        elif claim.identity != import_identity or claim.generation != import_generation:
            raise ValueError("external import claim does not match the requested generation")
        run_pk = claim.run_id
        scheduled_at = claim.scheduled_at
        captured_at = claim.captured_at
        claimed_run = await writer.ensure_capture_run(
            run_pk,
            started_at=claim.started_at,
            dataset_id=dataset_id,
            dataset_sha256=run_dataset_sha256,
            scheduled_at=scheduled_at,
            persona_id=coval_run.persona_id or None,
        )

        # Grouped as built: metric_type is a plain str on Result, so filtering the
        # flat list by enum identity would silently match nothing.
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
        all_rows = [row for rows in by_metric.values() for row in rows]
        rows = by_metric.get(Metric.V2V, [])
        capture_outcomes: list[Any] = []
        expected_capture_ids: list[str] = []
        if all_rows:
            grouped: dict[str, list[Result]] = {}
            for row in all_rows:
                grouped.setdefault(cast("str", row.audio_filename), []).append(row)
            dataset_fingerprint = fingerprint_for(run_dataset_sha256)
            from coval_bench.runner.capture import (
                RunManifest,
                build_capture_identity,
                identity_digest,
                upload_run_state,
            )
            from coval_bench.runner.persistence import persist_capture, prepare_capture_envelope

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
            for index, (sample_id, sample_rows) in enumerate(sorted(grouped.items())):
                sample_captured_at = captured_at + timedelta(microseconds=index)
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
                    captured_at=sample_captured_at,
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
        logger.info(
            "fetched_clips",
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            slot=str(scheduled_at),
            clips=len(rows),
            instruction=len(by_metric.get(Metric.INSTRUCTION_FOLLOWING, [])),
            workflow=len(by_metric.get(Metric.WORKFLOW_ADHERENCE, [])),
            ttft=len(by_metric.get(Metric.TTFT, [])),
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
