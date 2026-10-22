# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch benchmark metrics from the Coval API and write per-conversation rows.

Ingests each provider's recent completed runs not yet in the DB, slotted by
run create_time, and flags providers with no fresh data. Agent ids, the
metric id, and the API key are read from the environment, never committed.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

import click
import httpx
import structlog
from google.cloud import storage

from coval_bench import scenarios
from coval_bench.config import Settings, get_settings, require_capture_bucket
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.models import (
    RunStatus,
)
from coval_bench.db.registry_store import fetch_models
from coval_bench.db.writer import RunWriter
from coval_bench.registries import Metric
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.conditions import (
    DATASET_ID,
    Condition,
    condition_for,
    dataset_id_for,
)
from coval_bench.s2s.coval_runs import (
    WINDOW_PAGE_SIZE,
    CovalRun,
    bucket_start,
    coval_client,
    recent_completed_runs,
)
from coval_bench.s2s.identity import (
    UnmappedPersona,
    all_persona_conditions,
    all_persona_labels,
    dataset_identity,
    fetches_v2v,
    packaged_dataset_sha256,
    parse_persona_conditions,
)
from coval_bench.s2s.ingest import (
    import_identity,
    ingest_run,
    pending_metrics,
    resolve_import_generation,
)
from coval_bench.s2s.rows import VALUE_MAPPERS, ingestable_metrics
from coval_bench.s2s.samples import SampleRun, model_labels, publish_tick_sample
from coval_bench.s2s.specs import AgentSpec, llm_specs, s2s_specs

logger = structlog.get_logger("coval_bench.s2s.fetch_v2v")


def _backfill_status(statuses: list[RunStatus]) -> RunStatus:
    """A targeted backfill's verdict, which freshness is no part of.

    Nothing matched is a success here: the named runs belong to one agent, so every
    other provider legitimately has nothing to do.
    """
    if not statuses:
        return RunStatus.SUCCEEDED
    if all(s is RunStatus.FAILED for s in statuses):
        return RunStatus.FAILED
    if all(s is RunStatus.SUCCEEDED for s in statuses):
        return RunStatus.SUCCEEDED
    return RunStatus.PARTIAL


def _expected_sample_models(settings: Settings, dataset_id: str) -> set[tuple[str, str]]:
    """The pairs a sample on *dataset_id* must cover; a bucket short one publishes nothing.

    Each dataset is its own samples partition, so only the agents whose clean
    condition lands there count; a model on another set is not missing.
    """
    return {
        (spec.provider, spec.model)
        for spec in s2s_specs(settings)
        if spec.agent_id
        and spec.publish_samples
        and dataset_id_for(spec.family, Condition.CLEAN) == dataset_id
    }


async def _fetch_one_provider(
    client: httpx.AsyncClient,
    writer: RunWriter,
    *,
    spec: AgentSpec,
    agent_id: str,
    metric_ids: Mapping[Metric, str],
    test_set_id: str | None = None,
    noisy_persona_id: str | None = None,
    persona_conditions: Mapping[str, Condition] | None = None,
    period_seconds: int,
    stale_grace_seconds: int,
    sampled_runs: list[SampleRun] | None = None,
    only_run_ids: frozenset[str] | None = None,
    window_seconds: int | None = None,
    page_size: int = WINDOW_PAGE_SIZE,
    matched_run_ids: set[str] | None = None,
    artifact_client: storage.Client | None = None,
    artifact_bucket: str = "",
    workspace_id: str | None = None,
    unmapped_personas: dict[str, int] | None = None,
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
    # The clean caller, plus the legacy single-turn manifest which has no conditions.
    publishable = {DATASET_ID, dataset_id_for(spec.family, Condition.CLEAN)}

    def note_sample_candidate(coval_run: CovalRun, dataset_id: str) -> None:
        # Newest-first scan: keep the newest eligible run PER (BUCKET, PERSONA).
        # Multi-turn runs the same test set once per persona, so a provider yields
        # one candidate per persona per day (single-turn has no persona -> a "" key).
        # Keying on the bucket too is what lets a missed day still be published: the
        # window spans two periods, and keying on persona alone let today's run evict
        # yesterday's before the sampler ever saw it. Staggered arrivals must not
        # shrink the sample; committed only after the staleness check so a stale
        # provider's old recording never ships today.
        # Every other condition's recording stays internal, as does every recording
        # from an agent under embargo.
        if only_run_ids is not None:
            return
        if not spec.publish_samples or dataset_id not in publishable:
            return
        bucket_at = bucket_start(coval_run.create_time or datetime.now(tz=UTC), period_seconds)
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
            test_set_id=test_set_id or "",
            dataset_id=dataset_id,
        )

    try:
        runs = await recent_completed_runs(
            client,
            agent_id,
            period_seconds=period_seconds,
            test_set_id=test_set_id,
            window_seconds=window_seconds,
            page_size=page_size,
            requested_run_ids=only_run_ids,
            workspace_id=workspace_id,
        )
        data_seen = False
        newest_data_at: datetime | None = None
        for coval_run in runs:
            if only_run_ids is not None and coval_run.run_id not in only_run_ids:
                continue
            # An unmapped persona skips only its own run: the provider's other runs
            # still ingest, and the job reports the persona once at the end.
            try:
                identity = dataset_identity(
                    test_set_id,
                    coval_run.persona_id,
                    noisy_persona_id,
                    family=spec.family,
                    persona_conditions=persona_conditions,
                )
            except UnmappedPersona:
                logger.warning(
                    "run_persona_unmapped",
                    provider=spec.provider,
                    coval_run_id=coval_run.run_id,
                    persona_id=coval_run.persona_id,
                )
                if unmapped_personas is not None:
                    unmapped_personas[coval_run.persona_id] = (
                        unmapped_personas.get(coval_run.persona_id, 0) + 1
                    )
                continue
            if identity is None:
                continue
            dataset_id, dataset_sha256 = identity
            condition = condition_for(dataset_id)
            if condition.benchmark is not spec.benchmark:
                raise RuntimeError(
                    f"dataset {dataset_id!r} belongs to {condition.benchmark}, not {spec.benchmark}"
                )
            if condition.required not in metric_ids:
                logger.warning(
                    "required_metric_unconfigured",
                    provider=spec.provider,
                    dataset_id=dataset_id,
                    metric=condition.required.value,
                )
                continue
            ingestable = ingestable_metrics(condition, metric_ids)
            if artifact_client is None:
                raise RuntimeError("required capture storage is unavailable")
            import_identity_value = import_identity(
                spec=spec,
                coval_run=coval_run,
                dataset_id=dataset_id,
                dataset_sha256=dataset_sha256 or packaged_dataset_sha256(),
                workspace_id=workspace_id,
            )
            import_claim, import_generation = await resolve_import_generation(
                artifact_client, artifact_bucket, import_identity_value
            )
            database_pending = await pending_metrics(
                writer,
                benchmark=spec.benchmark,
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                condition=condition,
                metric_ids=metric_ids,
            )
            persisted = ingestable - database_pending
            if persisted:
                note_sample_candidate(coval_run, dataset_id)
            if condition.required in persisted and not data_seen:
                data_seen, newest_data_at = True, coval_run.create_time
            if import_generation is None:
                logger.warning(
                    "normalized_import_recovery_pending",
                    provider=spec.provider,
                    coval_run_id=coval_run.run_id,
                )
                continue
            pending = (
                frozenset(Metric(metric) for metric in import_claim.metric_types)
                if import_claim is not None
                else database_pending
            )
            if not pending:
                if matched_run_ids is not None and only_run_ids is not None:
                    matched_run_ids.add(coval_run.run_id)
                logger.info(
                    "run_already_ingested", provider=spec.provider, coval_run_id=coval_run.run_id
                )
                continue
            status = await ingest_run(
                client,
                writer,
                spec=spec,
                coval_run=coval_run,
                metric_ids=metric_ids,
                condition=condition,
                pending=pending,
                dataset_id=dataset_id,
                dataset_sha256=dataset_sha256,
                period_seconds=period_seconds,
                artifact_client=artifact_client,
                artifact_bucket=artifact_bucket,
                workspace_id=workspace_id,
                import_identity=import_identity_value,
                import_generation=import_generation,
                import_claim=import_claim,
            )
            if status is None:
                continue
            if (
                status is not RunStatus.FAILED
                and matched_run_ids is not None
                and only_run_ids is not None
            ):
                matched_run_ids.add(coval_run.run_id)
            if status is not RunStatus.FAILED:
                note_sample_candidate(coval_run, dataset_id)
                if not data_seen:
                    data_seen, newest_data_at = True, coval_run.create_time
            statuses.append(status)

        if only_run_ids is not None:
            return _backfill_status(statuses), len(statuses)

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


def _require_family_test_sets(settings: Settings, specs: Sequence[AgentSpec]) -> None:
    """A family's test set is required once one of its agents is configured.

    Unset would otherwise skip the agent with a warning that reads the same as
    never having configured it.
    """
    for spec in specs:
        if spec.test_set_id or not spec.test_set_id_attr or not spec.agent_id:
            continue
        if not (getattr(settings, spec.test_set_id_attr) or "").strip():
            raise RuntimeError(
                f"{spec.test_set_id_attr} is required when the {spec.provider} agent is set"
            )


def validate_fetch_settings(
    settings: Settings, specs: Sequence[AgentSpec], benchmark: Benchmark
) -> None:
    """Reject settings the fetch would fail on, before it reads Coval or the database."""
    if benchmark is Benchmark.S2S and not settings.coval_s2s_latency_metric_id:
        # Only a configured spec whose clean condition fetches V2V (see
        # conditions.py) makes the latency metric required.
        v2v_spec_configured = any(spec.agent_id and fetches_v2v(spec.family) for spec in specs)
        if v2v_spec_configured:
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
    if benchmark is Benchmark.LLM:
        unset = scenarios.ACTIVE.missing(settings, benchmark=Benchmark.LLM)
        if unset:
            raise RuntimeError(f"{', '.join(unset)} is required for the LLM benchmark")
    if benchmark is Benchmark.S2S and bool(instruction_metric_id) != bool(test_set_id):
        raise RuntimeError(
            "coval_s2s_instruction_metric_id and coval_s2s_test_set_id must be set together"
        )
    # Blank rather than unset would skip its agents with only a warning, which
    # reads the same as never having configured them.
    raw_happypath = settings.coval_s2s_happypath_test_set_id
    if raw_happypath is not None and not raw_happypath.strip():
        raise RuntimeError("coval_s2s_happypath_test_set_id must not be blank")
    raw_dental = settings.coval_s2s_dental_test_set_id
    if raw_dental is not None and not raw_dental.strip():
        raise RuntimeError("coval_s2s_dental_test_set_id must not be blank")
    _require_family_test_sets(settings, specs)
    # The noisy persona only separates conditions within a test set, so without
    # one it would silently never take effect.
    raw_noisy = settings.coval_s2s_noisy_persona_id
    if raw_noisy is not None and not raw_noisy.strip():
        raise RuntimeError("coval_s2s_noisy_persona_id must not be blank")
    noisy_persona_id = raw_noisy or None
    if benchmark is Benchmark.S2S and noisy_persona_id and not test_set_id:
        raise RuntimeError("coval_s2s_noisy_persona_id requires coval_s2s_test_set_id")
    parse_persona_conditions(all_persona_conditions(settings))
    # Scenario personas travel with their scenario's own test set, so only the
    # legacy map needs the shared one.
    if benchmark is Benchmark.S2S and settings.coval_s2s_condition_personas and not test_set_id:
        raise RuntimeError("coval_s2s_condition_personas requires coval_s2s_test_set_id")
    # The text board reads bank's test set from its own setting; the two must agree
    # or voice and text silently measure different Coval sets.
    bank_ids = settings.coval_s2s_scenarios.get(scenarios.ACTIVE.slug)
    text_test_set = (settings.coval_s2s_bank_test_set_id or "").strip()
    if bank_ids and text_test_set and bank_ids.test_set_id != text_test_set:
        raise RuntimeError(
            f"coval_s2s_scenarios['bank'].test_set_id {bank_ids.test_set_id!r} differs from "
            f"coval_s2s_bank_test_set_id {text_test_set!r}"
        )
    raw_interruption = settings.coval_s2s_interruption_metric_id
    if raw_interruption is not None and not raw_interruption.strip():
        raise RuntimeError("coval_s2s_interruption_metric_id must not be blank")
    raw_call_length = settings.coval_s2s_call_length_metric_id
    if raw_call_length is not None and not raw_call_length.strip():
        raise RuntimeError("coval_s2s_call_length_metric_id must not be blank")


async def fetch_and_write_v2v(
    settings: Settings | None = None,
    *,
    benchmark: Benchmark = Benchmark.S2S,
    only_run_ids: frozenset[str] | None = None,
    window_seconds: int | None = None,
    page_size: int = WINDOW_PAGE_SIZE,
    llm_agent_ids: Mapping[str, str] | None = None,
) -> dict[str, RunStatus]:
    """Ingest the selected benchmark's providers; return per-provider status.

    Each ingested Coval run gets its own run row slotted by its create_time,
    and metric-aware, benchmark-specific dedup makes re-scans no-ops, so ticks
    are idempotent and the fetch cadence only affects how soon data appears —
    the cron may run more often than the sims.
    """
    settings = settings or get_settings()
    require_capture_bucket(settings)
    specs = s2s_specs(settings) if benchmark is Benchmark.S2S else ()

    validate_fetch_settings(settings, specs, benchmark)
    metric_id = settings.coval_s2s_latency_metric_id
    instruction_metric_id = settings.coval_s2s_instruction_metric_id or None
    test_set_id = settings.coval_s2s_test_set_id or None
    noisy_persona_id = settings.coval_s2s_noisy_persona_id or None
    # Supersedes coval_s2s_noisy_persona_id once set, so the two can deploy in
    # either order.
    persona_conditions = parse_persona_conditions(all_persona_conditions(settings))
    persona_labels = all_persona_labels(settings)
    raw_interruption = settings.coval_s2s_interruption_metric_id
    raw_call_length = settings.coval_s2s_call_length_metric_id
    # Only configured metrics are ever asked for, so an unset id simply means that
    # metric is not ingested yet.
    metric_ids: dict[Metric, str] = {}
    if metric_id:
        metric_ids[Metric.V2V] = metric_id
    if instruction_metric_id:
        metric_ids[Metric.INSTRUCTION_FOLLOWING] = instruction_metric_id
    if raw_interruption:
        metric_ids[Metric.INTERRUPTION_RATE] = raw_interruption
    if raw_call_length:
        metric_ids[Metric.CALL_LENGTH] = raw_call_length
    unmapped = sorted(m.value for m in metric_ids.keys() - VALUE_MAPPERS.keys())
    if unmapped:
        raise RuntimeError(f"no VALUE_MAPPERS entry for configured metrics: {', '.join(unmapped)}")

    async with coval_client(settings) as client, lifespan_pool(settings) as pool:
        models = await fetch_models(pool)
        if benchmark is Benchmark.LLM:
            specs = llm_specs(models, llm_agent_ids)
            _require_family_test_sets(settings, specs)
        registered = frozenset((m.provider, m.model) for m in models)
        writer = RunWriter(pool)
        from coval_bench.runner.capture import preflight_capture_storage

        artifact_bucket = settings.benchmark_artifact_bucket
        artifact_client = storage.Client()
        await asyncio.to_thread(preflight_capture_storage, artifact_client, artifact_bucket)
        await writer.preflight_required_capture_schema()
        statuses: dict[str, RunStatus] = {}
        total_ingested = 0
        matched_run_ids: set[str] = set()
        unmapped_personas: dict[str, int] = {}
        sampled_runs: list[SampleRun] = []
        for spec in specs:
            agent_id = spec.agent_id
            if not agent_id:
                logger.warning("agent_id_unset", provider=spec.provider, model=spec.model)
                continue
            if (spec.provider, spec.model) not in registered:
                logger.error(
                    "model_unregistered",
                    provider=spec.provider,
                    model=spec.model,
                    fix="add the registry row (published=false) before enabling this agent",
                )
                continue
            # An agent on its own test set is skipped when that id is missing rather
            # than falling back to the shared one, which would file its runs under
            # the wrong family.
            spec_test_set = (
                spec.test_set_id
                or (getattr(settings, spec.test_set_id_attr) if spec.test_set_id_attr else None)
                or (None if spec.test_set_id_attr else test_set_id)
            ) or None
            if spec.test_set_id_attr and not spec_test_set:
                logger.warning("test_set_unset", provider=spec.provider, attr=spec.test_set_id_attr)
                continue
            spec_workspace_id = (
                (getattr(settings, spec.workspace_id_attr) or "").strip() or None
                if spec.workspace_id_attr
                else None
            )
            if spec.workspace_id_attr and not spec_workspace_id:
                logger.warning(
                    "workspace_id_unset", provider=spec.provider, attr=spec.workspace_id_attr
                )
                continue
            spec_metric_ids = metric_ids
            if spec.instruction_metric_id_attr:
                spec_instruction_metric_id = (
                    getattr(settings, spec.instruction_metric_id_attr) or ""
                ).strip()
                if not spec_instruction_metric_id:
                    logger.warning(
                        "instruction_metric_id_unset",
                        provider=spec.provider,
                        attr=spec.instruction_metric_id_attr,
                    )
                    continue
                spec_metric_ids = {
                    **metric_ids,
                    Metric.INSTRUCTION_FOLLOWING: spec_instruction_metric_id,
                }
            spec_metric_ids = {**spec_metric_ids, **spec.metric_ids}
            status_key = f"{spec.family}:{spec.provider}:{spec.model}"
            statuses[status_key], ingested = await _fetch_one_provider(
                client,
                writer,
                spec=spec,
                agent_id=agent_id,
                metric_ids=spec_metric_ids,
                test_set_id=spec_test_set,
                noisy_persona_id=noisy_persona_id,
                persona_conditions=persona_conditions,
                period_seconds=settings.s2s_fetch_period_seconds,
                stale_grace_seconds=settings.s2s_stale_grace_seconds,
                sampled_runs=sampled_runs,
                only_run_ids=only_run_ids,
                window_seconds=window_seconds,
                page_size=page_size,
                matched_run_ids=matched_run_ids,
                artifact_client=artifact_client,
                artifact_bucket=artifact_bucket,
                workspace_id=spec_workspace_id,
                unmapped_personas=unmapped_personas,
            )
            total_ingested += ingested

        if unmapped_personas:
            from coval_bench.logging import log_run_unmapped_persona

            log_run_unmapped_persona(unmapped_personas)

        try:
            await writer.refresh_window_views()
        except Exception:
            logger.warning("dashboard_windows_refresh_failed", exc_info=True)

        if only_run_ids is not None and (unmatched := only_run_ids - matched_run_ids):
            logger.error("backfill_runs_not_found", coval_run_ids=sorted(unmatched))
            raise RuntimeError(
                f"targeted backfill did not recover runs: {', '.join(sorted(unmatched))}"
            )

        # One sample per dataset: each family runs its own test set, so a tick
        # that pooled them would never find a scenario shared by every model. The
        # set the recordings actually came from labels the manifest; gating on the
        # shared ``coval_s2s_test_set_id`` instead would stop publishing entirely
        # once every agent carries its own set.
        by_dataset: dict[str, list[SampleRun]] = {}
        for run in sampled_runs:
            by_dataset.setdefault(run.dataset_id, []).append(run)
        for dataset_id, dataset_runs in sorted(by_dataset.items()):
            sample_test_set_id = next(
                (r.test_set_id for r in dataset_runs if r.test_set_id), test_set_id
            )
            if not (settings.s2s_samples_bucket and sample_test_set_id):
                continue
            expected = _expected_sample_models(settings, dataset_id)
            missing = expected - {r.key for r in dataset_runs}
            if missing:
                # Error level on purpose: this is the alert that a model is
                # absent from the window, so no day in it can publish a sample.
                logger.error(
                    "samples_provider_missing", dataset=dataset_id, missing=model_labels(missing)
                )
            await publish_tick_sample(
                client,
                bucket_name=settings.s2s_samples_bucket,
                dataset_id=dataset_id,
                test_set_id=sample_test_set_id,
                runs=dataset_runs,
                rng=random.Random(),  # noqa: S311
                expected_models=expected,
                persona_labels=persona_labels,
            )
        logger.info(
            "s2s_fetch_done",
            statuses={p: str(s) for p, s in statuses.items()},
            ingested=total_ingested,
        )
        return statuses


def _fetch_options[Command: Callable[..., None]](command: Command) -> Command:
    options = (
        click.option(
            "--coval-run-id",
            "coval_run_ids",
            multiple=True,
            help="Ingest only these Coval runs, scanning beyond the scheduled window. Repeatable.",
        ),
        click.option(
            "--window-hours",
            type=click.IntRange(min=1),
            default=720,
            show_default=True,
            help="How far back --coval-run-id searches. Ignored without it.",
        ),
        click.option(
            "--page-size",
            type=click.IntRange(min=1),
            default=100,
            show_default=True,
            help="Runs listed per agent while searching for --coval-run-id. Ignored without it.",
        ),
    )
    for option in reversed(options):
        command = option(command)
    return command


def _run_fetch(
    benchmark: Benchmark,
    coval_run_ids: tuple[str, ...],
    window_hours: int,
    page_size: int,
    *,
    settings: Settings | None = None,
    llm_agent_ids: Mapping[str, str] | None = None,
) -> None:
    from coval_bench.logging import configure_logging, log_run_failed, log_run_partial

    label = f"{benchmark.value.lower()} fetch"
    settings = settings or get_settings()
    configure_logging(level=settings.log_level)
    # A setup crash fails the whole job.
    try:
        only_run_ids = frozenset(coval_run_ids) or None
        statuses = asyncio.run(
            fetch_and_write_v2v(
                settings,
                benchmark=benchmark,
                only_run_ids=only_run_ids,
                window_seconds=window_hours * 3600 if only_run_ids else None,
                page_size=page_size if only_run_ids else WINDOW_PAGE_SIZE,
                llm_agent_ids=llm_agent_ids,
            )
        )
    except Exception as exc:
        log_run_failed(str(exc), exc)
        raise

    # Healthy providers' rows are already committed, so a PARTIAL run alerts but
    # still exits 0 (no Cloud Run retry). Only a total loss fails the job.
    failed = [p for p, s in statuses.items() if s is RunStatus.FAILED]
    if not statuses or all(s is RunStatus.FAILED for s in statuses.values()):
        if statuses:
            log_run_failed(f"{label} failed for all providers: {', '.join(failed)}")
        else:
            log_run_failed(f"{label} ran no providers (none configured)")
        raise click.ClickException(f"{label} failed for all providers")
    if failed:
        log_run_partial(f"{label} has no fresh data from: {', '.join(failed)}")


@click.command(name="fetch-s2s")
@_fetch_options
def fetch_s2s(coval_run_ids: tuple[str, ...], window_hours: int, page_size: int) -> None:
    """Fetch S2S latency from Coval and write per-clip rows (scheduled Cloud Run Job).

    With --coval-run-id it becomes a targeted backfill instead: only the named runs
    are ingested, no samples are published, and staleness is not judged, so an
    operator can recover runs the scheduled window has already passed over.
    """
    _run_fetch(Benchmark.S2S, coval_run_ids, window_hours, page_size)


if __name__ == "__main__":
    fetch_s2s()
