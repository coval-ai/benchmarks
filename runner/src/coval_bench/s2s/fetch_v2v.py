# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch benchmark metrics from the Coval API and write per-conversation rows.

Ingests each provider's recent completed runs not yet in the DB, slotted by
run create_time, and flags providers with no fresh data. Agent ids, the
metric id, and the API key are read from the environment, never committed.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.resources
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import click
import httpx
import structlog

from coval_bench.config import Settings, get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.models import MetricExecutor, Result, ResultStatus, RunStatus
from coval_bench.db.writer import RunWriter
from coval_bench.llm.phonely import MODEL as PHONELY_MODEL
from coval_bench.llm.phonely import PROVIDER as PHONELY_PROVIDER
from coval_bench.registries import METRIC_SPECS, Metric
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.conditions import (
    DATASET_ID,
    DEFAULT_CONDITION,
    FAMILY_DENTAL,
    FAMILY_LLM_DENTAL,
    FAMILY_MULTITURN,
    Condition,
    DatasetMetrics,
    condition_for,
    dataset_id_for,
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
    """One provider: the Settings attr holding its Coval agent id + display strings."""

    agent_id_attr: str
    provider: str
    model: str
    # Settings attr holding this agent's own Coval test set id; None uses the
    # shared ``coval_s2s_test_set_id``.
    test_set_id_attr: str | None = None
    # Dataset family for this agent's rows (see ``s2s.conditions``).
    family: str = FAMILY_MULTITURN
    # Whether this agent's recordings may reach the public samples card.
    publish_samples: bool = True
    benchmark: Benchmark = Benchmark.S2S


@dataclass(frozen=True)
class CovalRun:
    """One completed Coval run from the list endpoint's summary view."""

    run_id: str
    create_time: datetime | None
    persona_id: str = ""


# Agent ids resolved from Settings; model strings are display labels. Every agent
# now runs the dental test set: the shared multi-turn set stays queryable but is
# no longer written to, so a spec left on it would go stale rather than idle.
AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        agent_id_attr="coval_s2s_openai_agent_id",
        provider="openai",
        model="gpt-realtime",
        test_set_id_attr="coval_s2s_dental_test_set_id",
        family=FAMILY_DENTAL,
    ),
    AgentSpec(
        agent_id_attr="coval_s2s_gemini_agent_id",
        provider="google",
        model="gemini-live",
        test_set_id_attr="coval_s2s_dental_test_set_id",
        family=FAMILY_DENTAL,
    ),
    AgentSpec(
        agent_id_attr="coval_s2s_xai_agent_id",
        provider="xai",
        model="grok-voice-think-fast-1.0",
        test_set_id_attr="coval_s2s_dental_test_set_id",
        family=FAMILY_DENTAL,
    ),
    AgentSpec(
        agent_id_attr="coval_s2s_xai_think_fast_2_agent_id",
        provider="xai",
        model="grok-voice-think-fast-2.0",
        test_set_id_attr="coval_s2s_dental_test_set_id",
        family=FAMILY_DENTAL,
    ),
    # Pre-launch models under codenames: the provider and model strings are what
    # land in the results table, so they carry no vendor identity of their own.
    AgentSpec(
        agent_id_attr="coval_s2s_gray_agent_id",
        provider="colors",
        model="gray",
        test_set_id_attr="coval_s2s_dental_test_set_id",
        family=FAMILY_DENTAL,
        publish_samples=False,
    ),
    AgentSpec(
        agent_id_attr="coval_s2s_red_agent_id",
        provider="colors",
        model="red",
        test_set_id_attr="coval_s2s_dental_test_set_id",
        family=FAMILY_DENTAL,
        publish_samples=False,
    ),
    # The same dental set driven over text through the LLM proxy; TTFT comes from
    # the proxy's own turn log rather than from Coval.
    AgentSpec(
        agent_id_attr="coval_llm_phonely_agent_id",
        provider=PHONELY_PROVIDER,
        model=PHONELY_MODEL,
        test_set_id_attr="coval_s2s_dental_test_set_id",
        family=FAMILY_LLM_DENTAL,
        publish_samples=False,
        benchmark=Benchmark.LLM,
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


def _normalized_dataset_sha256(provenance: str) -> str:
    """Return the normalized schema stable 64-hex dataset fingerprint.

    Packaged datasets already carry a content SHA. Coval-hosted S2S datasets
    carry immutable test-set/persona provenance in the legacy run column, so
    fingerprint that identifier without changing the legacy representation.
    """
    if len(provenance) == 64 and all(character in "0123456789abcdef" for character in provenance):
        return provenance
    return hashlib.sha256(provenance.encode()).hexdigest()


def _condition_for_persona(
    persona_id: str,
    persona_conditions: Mapping[str, Condition] | None,
    noisy_persona_id: str | None,
) -> Condition:
    """This run's caller condition.

    An unmapped persona raises rather than counting as clean: that pooling would be
    invisible in both the data and the logs. ``noisy_persona_id`` is the pre-map
    setting, honoured only while the map is unset so the two can deploy in either
    order.
    """
    if persona_conditions:
        try:
            return persona_conditions[persona_id]
        except KeyError:
            raise ValueError(f"persona {persona_id!r} has no condition mapped") from None
    if noisy_persona_id and persona_id == noisy_persona_id:
        return Condition.NOISY
    return Condition.CLEAN


def _dataset_identity(
    test_set_id: str | None,
    persona_id: str = "",
    noisy_persona_id: str | None = None,
    *,
    family: str = FAMILY_MULTITURN,
    persona_conditions: Mapping[str, Condition] | None = None,
) -> tuple[str, str] | None:
    """Dataset id + provenance for one run's rows, or None when not ingested.

    Multi-turn runs are keyed to their Coval test set, not the single-turn SLURP
    manifest, so they get their own dataset id (never pooling with s2s-v1) and
    record the test-set id as provenance. Without a test set (legacy latency-only
    mode) the rows stay under the packaged s2s-v1 manifest.

    Non-clean conditions name the persona in their provenance so the rows stay
    self-describing.
    """
    if not test_set_id:
        return DATASET_ID, _dataset_sha256()
    condition = _condition_for_persona(persona_id, persona_conditions, noisy_persona_id)
    dataset_id = dataset_id_for(family, condition)
    if dataset_id is None:
        return None
    if condition is Condition.CLEAN:
        return dataset_id, test_set_id
    return dataset_id, f"{test_set_id}:{persona_id}"


def _persona_conditions(raw: Mapping[str, str]) -> dict[str, Condition] | None:
    """Parse the persona -> condition setting, rejecting unknown condition names."""
    if not raw:
        return None
    parsed: dict[str, Condition] = {}
    for persona_id, name in raw.items():
        try:
            parsed[persona_id] = Condition(name)
        except ValueError:
            valid = ", ".join(c.value for c in Condition)
            raise RuntimeError(
                f"coval_s2s_condition_personas[{persona_id!r}] is {name!r}; expected one of {valid}"
            ) from None
    return parsed


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
    """Completed Coval runs for one agent within the ingest window, newest first.

    List returns the summary view newest-first; filter by agent_id + status
    (tags are not filterable). ``test_set_id`` narrows to one test set so other
    sims on the same agents (e.g. the single-turn set) are not ingested. Runs
    without a parseable create_time are kept: better to ingest with a fetch-time
    slot than to drop data.
    """
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
                persona_id=cast("str", r.get("persona_id") or ""),
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

    Ids, not counts: equal counts can be different conversations. Values with no id
    are excluded from both sides — two of them are not the same conversation, so
    matching them would invent an overlap. The caller trims to the overlap; only
    duplicates skip the metric for the run.
    """
    other_set = {sid for v in other_values if (sid := v.get("simulation_output_id"))}
    anchor_set = {sid for v in anchor_values if (sid := v.get("simulation_output_id"))}
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


def _interruption_value(raw: object) -> tuple[float | None, ResultStatus] | None:
    """Interruptions per minute, stored as-is; a clip with no numeric value becomes a FAILED row."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return round(float(raw), 2), ResultStatus.SUCCESS
    return None, ResultStatus.FAILED


# The only per-metric part: one raw Coval value -> (row value, status), or None to
# write no row. A metric is ingestable once it appears here.
_VALUE_MAPPERS: dict[Metric, Callable[[object], tuple[float | None, ResultStatus] | None]] = {
    Metric.V2V: _v2v_value,
    Metric.INSTRUCTION_FOLLOWING: _instruction_value,
    Metric.INTERRUPTION_RATE: _interruption_value,
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
                benchmark=spec.benchmark,
                metric_type=metric,
                metric_units=METRIC_SPECS[metric].units,
                metric_value=metric_value,
                audio_filename=f"{coval_run_id}/{sim_id}" if sim_id else f"{coval_run_id}/{i}",
                status=status,
            )
        )
    return rows


def _failed_conversation_rows(
    output_ids: list[str],
    anchor_values: list[dict[str, Any]],
    *,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """A FAILED latency row for each conversation with no anchor value.

    Coval omits a failed conversation from every metric's values but still lists
    it in ``results.output_ids``. Without these rows a model is judged only on
    the calls it survived, and a run's failures would not show in its success
    rate.

    An anchor value without a simulation_output_id (kept by ``_s2s_rows`` under
    its index fallback) can't be reconciled against ``output_ids``, so coverage
    is undecidable for the whole run: synthesize nothing rather than stamp a
    measured conversation FAILED.
    """
    covered = {v.get("simulation_output_id") for v in anchor_values}
    if not all(covered):
        logger.warning(
            "anchor_ids_missing",
            provider=spec.provider,
            coval_run_id=coval_run_id,
            anchors_without_id=sum(1 for v in anchor_values if not v.get("simulation_output_id")),
        )
        return []
    return [
        Result(
            run_id=run_pk,
            provider=spec.provider,
            model=spec.model,
            benchmark=spec.benchmark,
            metric_type=Metric.V2V,
            metric_units=METRIC_SPECS[Metric.V2V].units,
            metric_value=None,
            audio_filename=f"{coval_run_id}/{sim_id}",
            status=ResultStatus.FAILED,
        )
        for sim_id in output_ids
        if sim_id not in covered
    ]


# Metrics a condition declares ``local`` are measured on our side and looked up
# by the anchor's conversation ids instead of fetched from Coval. A local metric
# is ingestable once it appears here, mirroring _VALUE_MAPPERS.
_LOCAL_SOURCES: dict[
    Metric, Callable[[RunWriter, Sequence[str]], Awaitable[Mapping[str, float]]]
] = {
    Metric.TTFT: lambda writer, sim_ids: writer.conversation_ttft(sim_ids),
}


def _local_rows(
    values: Mapping[str, float],
    *,
    metric: Metric,
    run_pk: int,
    coval_run_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """One SUCCESS row per conversation measured locally, keyed like the anchor rows."""
    return [
        Result(
            run_id=run_pk,
            provider=spec.provider,
            model=spec.model,
            benchmark=spec.benchmark,
            metric_type=metric,
            metric_units=METRIC_SPECS[metric].units,
            metric_value=round(value, 3),
            audio_filename=f"{coval_run_id}/{sim_id}",
            status=ResultStatus.SUCCESS,
        )
        for sim_id, value in values.items()
    ]


def _metric_values(metrics: dict[str, Any], metric_id: str | None) -> list[dict[str, Any]] | None:
    """The metric's per-conversation values, or None when it is not on the run."""
    payload = metrics.get(metric_id) if metric_id else None
    if payload is None:
        return None
    return cast("list[dict[str, Any]]", payload.get("values", []))


def _ingestable(condition: DatasetMetrics, metric_ids: Mapping[Metric, str]) -> frozenset[Metric]:
    """The condition's metrics that are configured and have a row builder."""
    fetched = frozenset(
        metric for metric in condition.fetched if metric in metric_ids and metric in _VALUE_MAPPERS
    )
    return fetched | (condition.local & frozenset(_LOCAL_SOURCES))


async def _pending_metrics(
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
    for metric in _ingestable(condition, metric_ids):
        if not await writer.coval_metric_ingested(
            benchmark=benchmark,
            provider=provider,
            coval_run_id=coval_run_id,
            metric_type=metric,
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
    period_seconds: int,
    normalized_dual_write_enabled: bool = False,
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
        pending = condition.fetched | (condition.local & frozenset(_LOCAL_SOURCES))
    run_pk: int | None = None
    try:
        resp = await client.get(f"/runs/{coval_run.run_id}")
        resp.raise_for_status()
        run = cast("dict[str, Any]", resp.json()["run"])
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
            # An id-less value cannot be reconciled against the anchor or deduped,
            # and its positional key is not stable across metrics, so it is
            # rejected before the populations are compared.
            values = [v for v in values if v.get("simulation_output_id")]
            mismatch = _population_mismatch(anchor_values, values)
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

        # Local metrics are looked up for the anchor's conversations. A metric with
        # no values at all stays pending for a later scan; once any conversation
        # has a value the run counts as ingested, and the ones still missing are a
        # permanent coverage gap, as an UNKNOWN verdict is for instruction.
        local: dict[Metric, Mapping[str, float]] = {}
        sim_ids = [sid for v in anchor_values if (sid := v.get("simulation_output_id"))]
        for metric in sorted(condition.local & pending):
            measured = await _LOCAL_SOURCES[metric](writer, sim_ids)
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

        scheduled_at = _bucket_start(coval_run.create_time or datetime.now(tz=UTC), period_seconds)
        run_row = await writer.start_run(
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
        for metric, local_values in local.items():
            by_metric[metric] = _local_rows(
                local_values, metric=metric, run_pk=run_pk, coval_run_id=coval_run.run_id, spec=spec
            )
        if condition.required is Metric.V2V and Metric.V2V in writable:
            output_ids = [
                s
                for s in cast("list[Any]", (run.get("results") or {}).get("output_ids") or [])
                if isinstance(s, str)
            ]
            by_metric[Metric.V2V].extend(
                _failed_conversation_rows(
                    output_ids,
                    anchor_values,
                    run_pk=run_pk,
                    coval_run_id=coval_run.run_id,
                    spec=spec,
                )
            )
        all_rows = [row for rows in by_metric.values() for row in rows]
        rows = by_metric.get(Metric.V2V, [])
        if all_rows:
            captured_at = datetime.now(UTC)
            await writer.record_results(all_rows, created_at=captured_at)
            if normalized_dual_write_enabled and spec.benchmark is Benchmark.S2S:
                from coval_bench.runner.normalized import dual_write

                grouped: dict[str, list[Result]] = {}
                for row in all_rows:
                    if row.audio_filename is None:  # pragma: no cover -- S2S rows set it
                        logger.warning(
                            "normalized_s2s_sample_id_missing",
                            provider=spec.provider,
                            coval_run_id=coval_run.run_id,
                        )
                        continue
                    grouped.setdefault(row.audio_filename, []).append(row)
                normalized_sha256 = _normalized_dataset_sha256(dataset_sha256 or _dataset_sha256())
                for sample_id, sample_rows in grouped.items():
                    provider_error = None
                    if not any(row.status is ResultStatus.SUCCESS for row in sample_rows):
                        provider_error = next(
                            (row.error for row in sample_rows if row.error),
                            "Coval conversation produced no successful metric value",
                        )
                    try:
                        await dual_write(
                            writer=writer,
                            storage_client=None,
                            bucket="",
                            run_id=run_pk,
                            dataset_id=dataset_id,
                            dataset_sha256=normalized_sha256,
                            sample_id=sample_id,
                            entry=spec,
                            benchmark=Benchmark.S2S,
                            results=sample_rows,
                            provider_error=provider_error,
                            captured_at=captured_at,
                            executor=MetricExecutor.COVAL_API,
                            db_retry_attempts=3,
                        )
                    except Exception:
                        logger.warning(
                            "normalized_s2s_dual_write_failed",
                            provider=spec.provider,
                            coval_run_id=coval_run.run_id,
                            sample_id=sample_id,
                            exc_info=True,
                        )
        logger.info(
            "fetched_clips",
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            slot=str(scheduled_at),
            clips=len(rows),
            instruction=len(by_metric.get(Metric.INSTRUCTION_FOLLOWING, [])),
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
        await writer.finish_run(run_pk, status=status)
        if status in (RunStatus.SUCCEEDED, RunStatus.PARTIAL):
            try:
                await writer.refresh_bucket(run_pk, period_seconds=period_seconds)
            except Exception:
                logger.warning("refresh_bucket_failed", provider=spec.provider, exc_info=True)
            try:
                await writer.refresh_metric_values_bucket(run_pk)
            except Exception:
                logger.warning(
                    "normalized_bucket_refresh_failed", provider=spec.provider, exc_info=True
                )
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


def _expected_sample_models(settings: Settings) -> set[tuple[str, str]]:
    """The pairs a sample must cover; a bucket short one publishes nothing."""
    return {
        (spec.provider, spec.model)
        for spec in AGENTS
        if getattr(settings, spec.agent_id_attr) and spec.publish_samples
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
    normalized_dual_write_enabled: bool = False,
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
            test_set_id=test_set_id or "",
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
        )
        data_seen = False
        newest_data_at: datetime | None = None
        for coval_run in runs:
            if only_run_ids is not None and coval_run.run_id not in only_run_ids:
                continue
            identity = _dataset_identity(
                test_set_id,
                coval_run.persona_id,
                noisy_persona_id,
                family=spec.family,
                persona_conditions=persona_conditions,
            )
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
            ingestable = _ingestable(condition, metric_ids)
            pending = await _pending_metrics(
                writer,
                benchmark=spec.benchmark,
                provider=spec.provider,
                coval_run_id=coval_run.run_id,
                condition=condition,
                metric_ids=metric_ids,
            )
            persisted = ingestable - pending
            if persisted:
                note_sample_candidate(coval_run, dataset_id)
            if condition.required in persisted and not data_seen:
                data_seen, newest_data_at = True, coval_run.create_time
            if not pending:
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
                metric_ids=metric_ids,
                condition=condition,
                pending=pending,
                dataset_id=dataset_id,
                dataset_sha256=dataset_sha256,
                period_seconds=period_seconds,
                normalized_dual_write_enabled=normalized_dual_write_enabled,
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


async def fetch_and_write_v2v(
    settings: Settings | None = None,
    *,
    benchmark: Benchmark = Benchmark.S2S,
    only_run_ids: frozenset[str] | None = None,
    window_seconds: int | None = None,
    page_size: int = WINDOW_PAGE_SIZE,
) -> dict[str, RunStatus]:
    """Ingest the selected benchmark's providers; return per-provider status.

    Each ingested Coval run gets its own run row slotted by its create_time,
    and ``coval_run_ingested`` makes re-scans no-ops, so ticks are idempotent
    and the fetch cadence only affects how soon data appears — the cron may
    run more often than the sims.
    """
    settings = settings or get_settings()
    specs = tuple(spec for spec in AGENTS if spec.benchmark is benchmark)

    metric_id = settings.coval_s2s_latency_metric_id
    if benchmark is Benchmark.S2S and not metric_id:
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
    if benchmark is Benchmark.LLM and not instruction_metric_id:
        raise RuntimeError("coval_s2s_instruction_metric_id is not set")
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
    # A family's test set is required once one of its agents is configured;
    # unset would otherwise skip the agent with a warning that reads the same
    # as never having configured it.
    for spec in specs:
        if not spec.test_set_id_attr or not getattr(settings, spec.agent_id_attr):
            continue
        if not (getattr(settings, spec.test_set_id_attr) or "").strip():
            raise RuntimeError(
                f"{spec.test_set_id_attr} is required when {spec.agent_id_attr} is set"
            )
    # The noisy persona only separates conditions within a test set, so without
    # one it would silently never take effect.
    raw_noisy = settings.coval_s2s_noisy_persona_id
    if raw_noisy is not None and not raw_noisy.strip():
        raise RuntimeError("coval_s2s_noisy_persona_id must not be blank")
    noisy_persona_id = raw_noisy or None
    if benchmark is Benchmark.S2S and noisy_persona_id and not test_set_id:
        raise RuntimeError("coval_s2s_noisy_persona_id requires coval_s2s_test_set_id")
    # Supersedes coval_s2s_noisy_persona_id once set, so the two can deploy in
    # either order.
    persona_conditions = _persona_conditions(settings.coval_s2s_condition_personas)
    if benchmark is Benchmark.S2S and persona_conditions and not test_set_id:
        raise RuntimeError("coval_s2s_condition_personas requires coval_s2s_test_set_id")
    raw_interruption = settings.coval_s2s_interruption_metric_id
    if raw_interruption is not None and not raw_interruption.strip():
        raise RuntimeError("coval_s2s_interruption_metric_id must not be blank")
    # Only configured metrics are ever asked for, so an unset id simply means that
    # metric is not ingested yet.
    metric_ids: dict[Metric, str] = {}
    if metric_id:
        metric_ids[Metric.V2V] = metric_id
    if instruction_metric_id:
        metric_ids[Metric.INSTRUCTION_FOLLOWING] = instruction_metric_id
    if raw_interruption:
        metric_ids[Metric.INTERRUPTION_RATE] = raw_interruption
    unmapped = sorted(m.value for m in metric_ids.keys() - _VALUE_MAPPERS.keys())
    if unmapped:
        raise RuntimeError(f"no _VALUE_MAPPERS entry for configured metrics: {', '.join(unmapped)}")

    async with _client(settings) as client, lifespan_pool(settings) as pool:
        writer = RunWriter(pool)
        statuses: dict[str, RunStatus] = {}
        total_ingested = 0
        matched_run_ids: set[str] = set()
        sampled_runs: list[SampleRun] = []
        for spec in specs:
            agent_id = getattr(settings, spec.agent_id_attr)
            if not agent_id:
                logger.warning("agent_id_unset", provider=spec.provider, attr=spec.agent_id_attr)
                continue
            # An agent on its own test set is skipped when that id is missing rather
            # than falling back to the shared one, which would file its runs under
            # the wrong family.
            spec_test_set = (
                getattr(settings, spec.test_set_id_attr) if spec.test_set_id_attr else test_set_id
            ) or None
            if spec.test_set_id_attr and not spec_test_set:
                logger.warning("test_set_unset", provider=spec.provider, attr=spec.test_set_id_attr)
                continue
            statuses[f"{spec.provider}:{spec.model}"], ingested = await _fetch_one_provider(
                client,
                writer,
                spec=spec,
                agent_id=agent_id,
                metric_ids=metric_ids,
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
                normalized_dual_write_enabled=settings.normalized_dual_write_enabled,
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

        # The set the sampled recordings actually came from, which is what labels
        # the manifest. Gating on the shared ``coval_s2s_test_set_id`` instead would
        # stop publishing entirely once every agent carries its own set, filling
        # sampled_runs and then never shipping them.
        sample_test_set_id = next(
            (r.test_set_id for r in sampled_runs if r.test_set_id), test_set_id
        )
        if settings.s2s_samples_bucket and sampled_runs and sample_test_set_id:
            expected = _expected_sample_models(settings)
            missing = expected - {r.key for r in sampled_runs}
            if missing:
                # Error level on purpose: this is the alert that a model is
                # absent from the window, so no day in it can publish a sample.
                logger.error("samples_provider_missing", missing=model_labels(missing))
            await publish_tick_sample(
                client,
                bucket_name=settings.s2s_samples_bucket,
                test_set_id=sample_test_set_id,
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
