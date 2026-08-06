# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Price collector: scrape pricing pages, LLM-extract rates, diff with a review gate.

Voice providers have no pricing APIs, so "dynamic" means: changes are detected
automatically, applied with provenance, and reviewed cheaply — never silently
published when large or uncertain. The gate per extracted rate:

* identical to the effective rate → noop
* delta ≤ 20% AND high extraction confidence AND (LiteLLM cross-check agrees
  or has no entry) → auto-insert a new effective row (``updated_by='bot'``,
  evidence = snapshot hash + verbatim quote), superseding the old one
* anything else (large delta, low confidence, cross-check disagreement, a
  model with no current rate) → a review item; the table is left untouched.

Review items go to Linear when ``LINEAR_API_KEY`` is set, else to the Slack
webhook, else they are logged. A staleness sweep flags active models whose
effective rate hasn't been re-verified in 45 days. One provider failing never
aborts the rest.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

import httpx
import structlog
from pydantic import BaseModel

from coval_bench.db.models import Benchmark, BillingUnit
from coval_bench.db.pricing import PricingStore
from coval_bench.registries.models import MODEL_REGISTRY, ModelStatus
from coval_bench.registries.pricing_sources import PRICING_SOURCES, PricingSource

if TYPE_CHECKING:
    import psycopg
    import psycopg.rows
    from psycopg_pool import AsyncConnectionPool

    from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.pricing")

_FETCH_TIMEOUT_S = 30.0
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_PAGE_TEXT_LIMIT = 120_000  # chars of page text handed to the extractor
_AUTO_APPLY_MAX_DELTA_PCT = 20.0
_STALE_AFTER_DAYS = 45
_LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

# Extracted page names → registry model ids, checked after an exact-id match
# fails. Keys are (provider, lowercased extracted name).
MODEL_ALIASES: dict[tuple[str, str], str] = {
    ("deepgram", "nova-3 monolingual"): "nova-3",
    ("deepgram", "nova-3 english"): "nova-3",
    ("deepgram", "flux english"): "flux-general-en",
    ("deepgram", "flux multilingual"): "flux-general-multi",
    ("deepgram", "aura-2"): "aura-2-thalia-en",
    ("assemblyai", "universal-streaming english"): "universal-streaming",
    ("assemblyai", "universal-streaming-english"): "universal-streaming",
    ("assemblyai", "universal-3.5 pro realtime"): "universal-3.5-pro",
    ("assemblyai", "u3-rt-pro"): "universal-3.5-pro",
    ("speechmatics", "real-time standard"): "default",
    ("speechmatics", "real-time enhanced"): "enhanced",
    ("elevenlabs", "flash / turbo"): "eleven_flash_v2_5",
    ("elevenlabs", "scribe v2 realtime"): "scribe_v2_realtime",
    ("revai", "reverb transcription"): "reverb",
    ("inworld", "realtime tts-2"): "inworld-tts-2",
    ("inworld", "realtime tts 1.5 max"): "inworld-tts-1.5-max",
    ("inworld", "realtime tts 1.5 mini"): "inworld-tts-1.5-mini",
    ("inworld", "stt 1"): "inworld-stt-1",
    ("rime", "mist v3"): "mistv3",
    ("together", "whisper large v3 (streaming)"): "whisper-large-v3",
    ("together", "nvidia nemotron 3.5 asr"): "nemotron-3.5-asr-streaming-0.6b",
    ("together", "nvidia parakeet tdt 0.6b v3"): "parakeet-tdt-0.6b-v3",
    ("xai", "speech to text"): "grok-stt",
    ("xai", "text to speech"): "grok-tts",
    ("modulate", "multilingual speech-to-text"): "velma-2-stt-streaming",
    ("modulate", "english fast speech-to-text"): "velma-2-stt-streaming-english-v2",
    ("smallest", "lightning v3.1 pro"): "lightning_v3.1_pro",
    ("smallest", "pulse realtime"): "pulse",
    ("mistral", "voxtral mini transcribe realtime"): "voxtral-mini-transcribe-realtime-2602",
}


class ExtractedRate(BaseModel):
    """One rate the LLM extracted, with its verbatim supporting quote."""

    model: str
    billing_unit: BillingUnit
    rate_usd: float
    plan: str | None = None
    confidence: Literal["high", "medium", "low"]
    quote: str
    note: str | None = None


class _Extraction(BaseModel):
    rates: list[ExtractedRate]


class Change(BaseModel):
    """Outcome of diffing one extracted rate against the effective rate."""

    provider: str
    model: str
    benchmark: Benchmark | None = None
    billing_unit: BillingUnit
    old_rate: Decimal | None = None
    new_rate: Decimal
    delta_pct: float | None = None
    action: Literal["noop", "auto_applied", "review", "unmatched"]
    reason: str = ""
    confidence: str = ""
    quote: str = ""
    snapshot_sha: str = ""
    source_url: str = ""


_EXTRACTION_PROMPT = """You extract published API list prices from a pricing page.

Rules — follow them exactly:
- Extract ONLY rates literally stated on the page. NEVER infer, estimate, or \
compute a rate that is not written there (converting a stated per-1k rate to \
per-1M by x1000 is allowed and must be explained in `note`).
- For each rate include `quote`: the EXACT verbatim page text stating it, \
copied character-for-character — no paraphrase, no added notes. Auto-apply is \
gated on this text being found in the page.
- billing_unit is the provider's native unit. Token-billed models produce two \
entries (input and output).
- When a promotional price shows a struck-through regular price, extract the \
current charged (promotional) price.
- confidence: "high" only when the model name and rate are unambiguous on the \
page; "medium" when the mapping needs interpretation; "low" otherwise.
- Speech/audio models only (STT, TTS, speech-to-speech); skip chat/LLM rates.

Provider: {provider}
Hints: {hints}
"""


async def _fetch_page(url: str) -> str:
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT_S, headers={"User-Agent": _USER_AGENT}, follow_redirects=True
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def _store_snapshot(
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    provider: str,
    url: str,
    text: str,
    *,
    dry_run: bool = False,
) -> tuple[str, bool]:
    """Persist the page (gzipped) unless its hash matches the latest snapshot.

    Returns ``(sha256, changed)`` — *changed* is False when the page is
    byte-identical to the previous fetch. ``dry_run`` computes the hash and
    compares, but never inserts.
    """
    sha = hashlib.sha256(text.encode()).hexdigest()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT sha256 FROM benchmarks_v2.pricing_snapshots"
                " WHERE provider = %s ORDER BY fetched_at DESC LIMIT 1",
                (provider,),
            )
            row = await cur.fetchone()
            if (row is not None and row["sha256"] == sha) or dry_run:
                await conn.commit()
                return sha, row is not None and row["sha256"] != sha
            await cur.execute(
                "INSERT INTO benchmarks_v2.pricing_snapshots (provider, url, sha256, content_gz)"
                " VALUES (%s, %s, %s, %s)",
                (provider, url, sha, gzip.compress(text.encode())),
            )
        await conn.commit()
    return sha, row is not None  # first-ever snapshot is not a "change"


async def _extract_rates(
    settings: Settings, provider: str, page_text: str, source: PricingSource
) -> list[ExtractedRate]:
    """LLM extraction with a strict JSON schema; quotes are mandatory."""
    import openai

    if settings.openai_api_key is None:
        raise RuntimeError("openai_api_key is required for price extraction")
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    completion = await client.chat.completions.parse(
        model=settings.pricing_extraction_model,
        messages=[
            {
                "role": "system",
                "content": _EXTRACTION_PROMPT.format(
                    provider=provider, hints=source.parse_hints or "none"
                ),
            },
            {"role": "user", "content": page_text[:_PAGE_TEXT_LIMIT]},
        ],
        response_format=_Extraction,
    )
    parsed = completion.choices[0].message.parsed
    return parsed.rates if parsed is not None else []


def _match_model(provider: str, extracted_name: str) -> tuple[str, Benchmark] | None:
    """Extracted page name → (registry model id, benchmark), or None.

    Exact id match first, then the alias map. A name matching a model id that
    the provider registers under more than one benchmark is ambiguous and
    treated as unmatched.
    """
    name = extracted_name.strip()
    model_id = MODEL_ALIASES.get((provider, name.lower()), name)
    entries = [m for m in MODEL_REGISTRY if m.provider == provider and m.model == model_id]
    benchmarks = {m.benchmark for m in entries}
    if len(benchmarks) != 1:
        return None
    return model_id, next(iter(benchmarks))


async def _litellm_prices() -> dict[str, Any]:
    """LiteLLM's community price file, or {} when unreachable (never fatal)."""
    try:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
            response = await client.get(_LITELLM_URL)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data
    except Exception:
        logger.warning("litellm_fetch_failed", exc_info=True)
        return {}


def _cross_check(
    provider: str,
    model: str,
    unit: BillingUnit,
    rate: Decimal,
    litellm: dict[str, Any],
) -> Literal["agree", "disagree", "unavailable"]:
    """Second-witness check against LiteLLM entries, where one exists.

    LiteLLM keys per-second audio as ``input_cost_per_second`` and token rates
    as ``*_cost_per_token``; entries are keyed ``model`` or ``provider/model``.
    """
    entry = litellm.get(f"{provider}/{model}") or litellm.get(model)
    if not isinstance(entry, dict):
        return "unavailable"
    ours_per_second: Decimal | None = None
    if unit is BillingUnit.PER_SECOND:
        ours_per_second = rate
    elif unit is BillingUnit.PER_MINUTE:
        ours_per_second = rate / 60
    elif unit is BillingUnit.PER_HOUR:
        ours_per_second = rate / 3600
    theirs: Any = None
    if ours_per_second is not None:
        theirs = entry.get("input_cost_per_second")
        ours: Decimal | None = ours_per_second
    elif unit is BillingUnit.PER_1M_TOKENS_INPUT:
        theirs = entry.get("input_cost_per_token")
        ours = rate / 1_000_000
    elif unit is BillingUnit.PER_1M_TOKENS_OUTPUT:
        theirs = entry.get("output_cost_per_token")
        ours = rate / 1_000_000
    else:
        return "unavailable"
    if theirs is None:
        return "unavailable"
    try:
        theirs_dec = Decimal(str(theirs))
    except ArithmeticError:
        return "unavailable"
    if theirs_dec == 0:
        return "unavailable"
    delta = abs(ours - theirs_dec) / theirs_dec * 100
    return "agree" if delta <= Decimal("5") else "disagree"


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _normalize_page_text(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).casefold().strip()


def _quote_in_page(quote: str, page: str) -> bool:
    """The auto-apply integrity gate: the extractor's verbatim quote must
    actually appear in the fetched page (tag-stripped, whitespace-collapsed),
    so a hallucinated or prompt-injected rate can never self-attest its way
    into the pricing table."""
    normalized_quote = _normalize_page_text(quote)
    return bool(normalized_quote) and normalized_quote in _normalize_page_text(page)


def _delta_pct(old: Decimal, new: Decimal) -> float:
    if old == 0:
        return float("inf") if new != 0 else 0.0
    return float(abs(new - old) / old * 100)


async def _diff_and_gate(
    store: PricingStore,
    extracted: ExtractedRate,
    *,
    provider: str,
    model: str,
    benchmark: Benchmark,
    source_url: str,
    snapshot_sha: str,
    page: str,
    litellm: dict[str, Any],
    dry_run: bool,
) -> Change:
    new_rate = Decimal(str(extracted.rate_usd))
    effective = [
        r
        for r in await store.get_effective_rates(provider, model, datetime.now(tz=UTC))
        if r.benchmark is benchmark and r.billing_unit is extracted.billing_unit
    ]
    current = effective[0] if effective else None
    change = Change(
        provider=provider,
        model=model,
        benchmark=benchmark,
        billing_unit=extracted.billing_unit,
        old_rate=current.rate_usd if current else None,
        new_rate=new_rate,
        confidence=extracted.confidence,
        quote=extracted.quote,
        snapshot_sha=snapshot_sha,
        source_url=source_url,
        action="noop",
    )
    if current is not None and current.rate_usd == new_rate:
        # Re-verified unchanged: stamp today's as_of so the rate never trips
        # the staleness alarm right after a successful check.
        if not dry_run and current.id is not None and current.as_of < date.today():  # noqa: DTZ011
            await store.refresh_as_of(current.id, date.today())  # noqa: DTZ011
        return change

    delta = _delta_pct(current.rate_usd, new_rate) if current is not None else None
    change.delta_pct = delta
    witness = _cross_check(provider, model, extracted.billing_unit, new_rate, litellm)

    if current is None:
        change.action = "review"
        change.reason = "no current effective rate — new rates always go to review"
        return change
    if delta is not None and delta > _AUTO_APPLY_MAX_DELTA_PCT:
        change.action = "review"
        change.reason = f"delta {delta:.1f}% exceeds {_AUTO_APPLY_MAX_DELTA_PCT:.0f}%"
        return change
    if extracted.confidence != "high":
        change.action = "review"
        change.reason = f"extraction confidence {extracted.confidence!r}"
        return change
    if witness == "disagree":
        change.action = "review"
        change.reason = "LiteLLM cross-check disagrees"
        return change
    if not _quote_in_page(extracted.quote, page):
        change.action = "review"
        change.reason = "extractor's quote not found verbatim in the fetched page"
        return change

    change.action = "auto_applied"
    change.reason = f"delta {delta:.1f}%, high confidence, cross-check {witness}"
    if not dry_run:
        await store.upsert_rate(
            provider=provider,
            model=model,
            benchmark=benchmark,
            billing_unit=extracted.billing_unit,
            rate_usd=new_rate,
            source_url=source_url,
            as_of=date.today(),  # noqa: DTZ011 — as_of is a calendar date
            evidence=f"snapshot sha256={snapshot_sha}; quote: {extracted.quote}",
            plan_assumption=extracted.plan,
            updated_by="bot",
        )
    return change


async def _stale_rates(store: PricingStore) -> list[str]:
    """Active models whose effective rate was last verified > 45 days ago."""
    await store.load_cache()
    today = date.today()  # noqa: DTZ011 — calendar-date comparison
    stale: list[str] = []
    for m in MODEL_REGISTRY:
        if m.status not in (ModelStatus.ACTIVE, ModelStatus.EARLY_ACCESS):
            continue
        for rate in store.effective_rates_cached(m.provider, m.model):
            if rate.benchmark is m.benchmark and (today - rate.as_of).days > _STALE_AFTER_DAYS:
                stale.append(
                    f"{m.benchmark}:{m.provider}/{m.model} [{rate.billing_unit}] "
                    f"as_of {rate.as_of} ({(today - rate.as_of).days}d old)"
                )
    return stale


async def _open_review(settings: Settings, title: str, body: str) -> None:
    """File one review item: Linear when configured, else Slack, else a log line."""
    if settings.linear_api_key is not None:
        query = """
        mutation IssueCreate($teamId: String!, $title: String!, $description: String!) {
          issueCreate(input: {teamId: $teamId, title: $title, description: $description}) {
            success
          }
        }
        """
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
            teams = await client.post(
                "https://api.linear.app/graphql",
                headers={"Authorization": settings.linear_api_key.get_secret_value()},
                json={
                    "query": (
                        "query Teams($name: String!) "
                        "{ teams(filter: {name: {eq: $name}}) { nodes { id } } }"
                    ),
                    "variables": {"name": settings.linear_team_name},
                },
            )
            teams.raise_for_status()
            nodes = teams.json()["data"]["teams"]["nodes"]
            if not nodes:
                raise RuntimeError(f"Linear team {settings.linear_team_name!r} not found")
            issue = await client.post(
                "https://api.linear.app/graphql",
                headers={"Authorization": settings.linear_api_key.get_secret_value()},
                json={
                    "query": query,
                    "variables": {"teamId": nodes[0]["id"], "title": title, "description": body},
                },
            )
            issue.raise_for_status()
        return
    if settings.pricing_review_slack_webhook is not None:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
            response = await client.post(
                settings.pricing_review_slack_webhook.get_secret_value(),
                json={"text": f"*{title}*\n{body}"},
            )
            response.raise_for_status()
        return
    logger.warning("pricing_review_item", title=title, body=body)


def _review_body(change: Change) -> str:
    return (
        f"provider/model: {change.provider}/{change.model} ({change.benchmark})\n"
        f"billing unit: {change.billing_unit}\n"
        f"current rate: {change.old_rate}\n"
        f"candidate rate: {change.new_rate}\n"
        f"delta: {f'{change.delta_pct:.1f}%' if change.delta_pct is not None else 'n/a (new)'}\n"
        f"reason: {change.reason}\n"
        f"confidence: {change.confidence}\n"
        f"quote: {change.quote}\n"
        f"snapshot: sha256={change.snapshot_sha}\n"
        f"source: {change.source_url}"
    )


async def collect_provider(
    settings: Settings,
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    store: PricingStore,
    provider: str,
    source: PricingSource,
    litellm: dict[str, Any],
    *,
    dry_run: bool,
) -> list[Change]:
    """Fetch → snapshot → extract → match → diff one provider. Raises on failure."""
    page = await _fetch_page(source.url)
    sha, _ = await _store_snapshot(pool, provider, source.url, page, dry_run=dry_run)
    extracted = await _extract_rates(settings, provider, page, source)

    changes: list[Change] = []
    for rate in extracted:
        matched = _match_model(provider, rate.model)
        if matched is None:
            logger.info("pricing_unmatched_model", provider=provider, extracted=rate.model)
            changes.append(
                Change(
                    provider=provider,
                    model=rate.model,
                    billing_unit=rate.billing_unit,
                    new_rate=Decimal(str(rate.rate_usd)),
                    action="unmatched",
                    reason="no registry model matches the extracted name",
                    confidence=rate.confidence,
                    quote=rate.quote,
                    snapshot_sha=sha,
                    source_url=source.url,
                )
            )
            continue
        model_id, benchmark = matched
        changes.append(
            await _diff_and_gate(
                store,
                rate,
                provider=provider,
                model=model_id,
                benchmark=benchmark,
                source_url=source.url,
                snapshot_sha=sha,
                page=page,
                litellm=litellm,
                dry_run=dry_run,
            )
        )
    return changes


async def run_update_prices(
    settings: Settings,
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    *,
    providers: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the collector across providers; one failure never aborts the rest."""
    store = PricingStore(pool)
    litellm = await _litellm_prices()
    selected = {
        name: source
        for name, source in PRICING_SOURCES.items()
        if source is not None and (providers is None or name in providers)
    }
    skipped = sorted(
        name
        for name, source in PRICING_SOURCES.items()
        if source is None and (providers is None or name in providers)
    )

    all_changes: list[Change] = []
    failed: list[str] = []
    for name, source in sorted(selected.items()):
        try:
            all_changes.extend(
                await collect_provider(
                    settings, pool, store, name, source, litellm, dry_run=dry_run
                )
            )
        except Exception:
            logger.warning("pricing_provider_failed", provider=name, exc_info=True)
            failed.append(name)

    reviews = [c for c in all_changes if c.action == "review"]
    if not dry_run:
        for change in reviews:
            try:
                await _open_review(
                    settings,
                    f"Price change review: {change.provider}/{change.model}",
                    _review_body(change),
                )
            except Exception:
                logger.warning(
                    "pricing_review_open_failed",
                    provider=change.provider,
                    model=change.model,
                    exc_info=True,
                )

    stale = await _stale_rates(store)
    alerts: list[str] = []
    if failed:
        alerts.append("page fetch/extract failed: " + ", ".join(failed))
    unmatched = [c for c in all_changes if c.action == "unmatched"]
    if unmatched:
        alerts.append(
            "unmatched page rates (possible new/renamed models, never auto-written): "
            + "; ".join(
                f"{c.provider}: {c.model!r} @ {c.new_rate} [{c.billing_unit}]" for c in unmatched
            )
        )
    if stale:
        alerts.append("stale rates (>45d): " + "; ".join(stale))
    if alerts and not dry_run:
        try:
            await _open_review(settings, "Pricing collector alerts", "\n".join(alerts))
        except Exception:
            logger.warning("pricing_alert_open_failed", exc_info=True)

    return {
        "changes": [json.loads(c.model_dump_json()) for c in all_changes],
        "counts": {
            action: sum(1 for c in all_changes if c.action == action)
            for action in ("noop", "auto_applied", "review", "unmatched")
        },
        "skipped_no_source": skipped,
        "failed_providers": failed,
        "stale_rates": stale,
        "dry_run": dry_run,
    }
