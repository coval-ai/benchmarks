# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Seed ``benchmarks_v2.model_pricing`` with researched published list rates.

One entry per ACTIVE / EARLY_ACCESS model in ``registries/models.py``, each in
the provider's native billing unit with the provider's public pricing page as
provenance. Subscription/credit providers are converted to an effective
per-unit rate under the Artificial-Analysis-style assumption recorded in
``plan_assumption``: the lowest-upfront plan that realistically supports
1,000 min/month (STT) or 1M chars/month (TTS).

Models whose price could not be verified on an official page get NO row —
they are printed as an explicit gap list instead (never guessed). The gap
list is enforced by ``scripts/check_pricing_coverage.py``.

Idempotent: re-running upserts nothing when the effective rates are unchanged
(``PricingStore.upsert_rate`` is a no-op on identical rate + plan). A changed
rate supersedes the old row, preserving history.

Run against a LOCAL/dev database only::

    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/benchmarks \\
        uv run python scripts/seed_pricing.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import NamedTuple
from urllib.parse import urlsplit

from coval_bench.config import get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.models import Benchmark, BillingUnit
from coval_bench.db.pricing import PricingStore
from coval_bench.registries.models import MODEL_REGISTRY, ModelStatus

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

#: Date the rates below were verified against the linked pricing pages.
AS_OF = date(2026, 8, 6)


class Rate(NamedTuple):
    """One published rate in the provider's native billing unit."""

    unit: BillingUnit
    rate_usd: str  # exact decimal string, e.g. "0.0059"
    source_url: str
    evidence: str
    plan_assumption: str | None = None


# Keyed by (benchmark, provider, model) — matching registries/models.py
# identifiers. Token-billed models carry two Rate entries (input + output).
# Every rate was verified on the linked page on AS_OF; unpriced models
# (enterprise/unpublished) are deliberately absent and surface as gaps.
RATESHEET: dict[tuple[Benchmark, str, str], tuple[Rate, ...]] = {
    # ------------------------------------------------------------------ STT
    (Benchmark.STT, "cartesia", "ink-2"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.049",
            "https://cartesia.ai/pricing",
            'Startup $49/mo includes "~115h 44m" STT; Pro $5/mo covers only ~9h16m',
            "Cartesia Startup plan $49/mo (1.25M credits, STT 3 credits/s); "
            "lowest plan supporting 1,000 min/mo: $49 / 1,000 min",
        ),
    ),
    (Benchmark.STT, "elevenlabs", "scribe_v2_realtime"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.39",
            "https://elevenlabs.io/pricing/api",
            'Scribe v2 Realtime — "$0.39 / Price per hour" (API billed in USD, not credits)',
        ),
    ),
    (Benchmark.STT, "soniox", "stt-rt-v5"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.12",
            "https://soniox.com/pricing",
            'stt-rt-v5 — "$0.12/hr" real-time transcription '
            "(underlying token billing: $2/1M audio-in + $4/1M text-out)",
        ),
    ),
    (Benchmark.STT, "smallest", "pulse"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.008",
            "https://docs.smallest.ai/models/model-cards/speech-to-text/pulse",
            'Pulse model card: "Realtime: $0.008/min" (batch $0.005/min)',
        ),
    ),
    (Benchmark.STT, "inworld", "inworld-stt-1"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.15",
            "https://inworld.ai/pricing",
            'STT On-Demand "$0.15/hr" (paid tiers discount to $0.10/hr)',
        ),
    ),
    (Benchmark.STT, "gradium", "default"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.013",
            "https://gradium.ai/pricing",
            '"Speech-to-Text: 3 credits / second"; XS $13/mo = 225k credits',
            "Gradium XS plan $13/mo (225k credits); 1,000 min = 180k credits "
            "fits in XS: $13 / 1,000 min",
        ),
    ),
    (Benchmark.STT, "openai", "gpt-4o-transcribe"): (
        Rate(
            BillingUnit.PER_1M_TOKENS_INPUT,
            "2.50",
            "https://developers.openai.com/api/docs/pricing",
            'audio input tokens: "gpt-4o-transcribe Transcription $2.50 $10.00 $0.006/minute"',
        ),
        Rate(
            BillingUnit.PER_1M_TOKENS_OUTPUT,
            "10.00",
            "https://developers.openai.com/api/docs/pricing",
            "text output tokens: gpt-4o-transcribe $10.00 per 1M",
        ),
    ),
    (Benchmark.STT, "openai", "gpt-4o-mini-transcribe"): (
        Rate(
            BillingUnit.PER_1M_TOKENS_INPUT,
            "1.25",
            "https://developers.openai.com/api/docs/pricing",
            'audio input tokens: "gpt-4o-mini-transcribe Transcription $1.25 $5.00 $0.003/minute"',
        ),
        Rate(
            BillingUnit.PER_1M_TOKENS_OUTPUT,
            "5.00",
            "https://developers.openai.com/api/docs/pricing",
            "text output tokens: gpt-4o-mini-transcribe $5.00 per 1M",
        ),
    ),
    (Benchmark.STT, "openai", "gpt-realtime-whisper"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.017",
            "https://developers.openai.com/api/docs/pricing",
            'Realtime models table: "gpt-realtime-whisper Audio - - $0.017 / minute"',
        ),
    ),
    (Benchmark.STT, "mistral", "voxtral-mini-transcribe-realtime-2602"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.006",
            "https://mistral.ai/news/voxtral-transcribe-2/",
            '"voxtral-mini-transcribe-realtime-2602 ... $0.006 per minute" '
            "(batch voxtral-mini-transcribe is $0.003/min)",
        ),
    ),
    (Benchmark.STT, "google", "chirp_2"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.016",
            "https://cloud.google.com/speech-to-text/pricing",
            "Speech-to-Text V2 Recognition (sku 3099-B70F-0949) Standard, "
            "0-500,000 min tier: $0.016/min; chirp models bill as Standard V2",
        ),
    ),
    (Benchmark.STT, "google", "chirp_3"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.016",
            "https://cloud.google.com/speech-to-text/pricing",
            "Speech-to-Text V2 Recognition Standard 0-500,000 min tier: $0.016/min",
        ),
    ),
    (Benchmark.STT, "together", "nemotron-3.5-asr-streaming-0.6b"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.0045",
            "https://www.together.ai/pricing",
            'per audio minute: "NVIDIA Nemotron 3.5 ASR $0.0045/min" (streaming)',
        ),
    ),
    (Benchmark.STT, "together", "parakeet-tdt-0.6b-v3"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.0015",
            "https://www.together.ai/pricing",
            'per audio minute: "NVIDIA Parakeet TDT 0.6B v3 $0.0015"',
        ),
    ),
    (Benchmark.STT, "together", "whisper-large-v3"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.0035",
            "https://www.together.ai/pricing",
            '"Whisper Large v3 (Streaming) $0.0035/min" (batch is $0.0015/min)',
        ),
    ),
    (Benchmark.STT, "xai", "grok-stt"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.20",
            "https://docs.x.ai/docs/models",
            'Voice pricing: "Speech to Text | $0.10/hr (REST), $0.20/hr (Streaming)" — streaming',
        ),
    ),
    (Benchmark.STT, "modulate", "velma-2-stt-streaming"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.06",
            "https://www.modulate.ai/api-pricing",
            '"Multilingual Speech-to-Text ... Streaming $0.06" per hour of audio',
        ),
    ),
    (Benchmark.STT, "modulate", "velma-2-stt-streaming-english-v2"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.05",
            "https://www.modulate.ai/api-pricing",
            '"English Fast Speech-to-Text ... Streaming $0.05" per hour of audio',
        ),
    ),
    (Benchmark.STT, "deepgram", "nova-2"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.35",
            "https://deepgram.com/pricing",
            'FAQ: "Nova-2 streaming at $0.35/hour" (legacy rate for existing deployments)',
        ),
    ),
    (Benchmark.STT, "deepgram", "nova-3"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.0048",
            "https://deepgram.com/pricing",
            'Nova-3 Monolingual streaming "$0.0048/min" (limited-time promo; regular list '
            "$0.0077/min shown struck through)",
        ),
    ),
    (Benchmark.STT, "deepgram", "flux-general-en"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.0065",
            "https://deepgram.com/pricing",
            'Flux English streaming "$0.0065/min" (promo; regular list $0.0077/min)',
        ),
    ),
    (Benchmark.STT, "deepgram", "flux-general-multi"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.0078",
            "https://deepgram.com/pricing",
            'Flux Multilingual streaming "$0.0078/min"',
        ),
    ),
    (Benchmark.STT, "assemblyai", "universal-streaming"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.15",
            "https://www.assemblyai.com/pricing",
            'Universal-Streaming English "$0.15/hr — billed on WebSocket session duration"',
        ),
    ),
    (Benchmark.STT, "assemblyai", "universal-streaming-multilingual"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.15",
            "https://www.assemblyai.com/pricing",
            'Universal-Streaming Multilingual "$0.15/hr" (session-duration billing)',
        ),
    ),
    (Benchmark.STT, "assemblyai", "universal-3.5-pro"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.45",
            "https://www.assemblyai.com/pricing",
            'Universal-3.5 Pro Realtime (u3-rt-pro) "$0.45/hr base" (session-duration billing)',
        ),
    ),
    (Benchmark.STT, "speechmatics", "default"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.45",
            "https://www.speechmatics.com/pricing",
            'Real-time Standard $0.45/hr, "billed to the second, based on the cost per hour"',
        ),
    ),
    (Benchmark.STT, "speechmatics", "enhanced"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.80",
            "https://www.speechmatics.com/pricing",
            "Real-time Enhanced $0.80/hr (billed to the second)",
        ),
    ),
    (Benchmark.STT, "revai", "reverb"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.20",
            "https://www.rev.ai/pricing",
            '"Reverb Transcription $0.20 per hour" (streaming billed on stream duration, '
            "15s minimum)",
        ),
    ),
    (Benchmark.STT, "gladia", "solaria-1"): (
        Rate(
            BillingUnit.PER_HOUR,
            "0.75",
            "https://www.gladia.io/pricing",
            'Starter (pay-as-you-go): "Real-time at $0.75/hr" (flat real-time rate, '
            "no per-model split)",
            "Gladia Starter prepaid PAYG plan; commit plans go as low as $0.25/hr",
        ),
    ),
    # ------------------------------------------------------------------ TTS
    (Benchmark.TTS, "inworld", "inworld-tts-2"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "25.00",
            "https://inworld.ai/pricing",
            "On-demand: Realtime TTS-2 $25 per 1M characters",
        ),
    ),
    (Benchmark.TTS, "inworld", "inworld-tts-1.5-max"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "35.00",
            "https://inworld.ai/pricing",
            "On-demand: Realtime TTS 1.5 Max $35 per 1M characters",
        ),
    ),
    (Benchmark.TTS, "inworld", "inworld-tts-1.5-mini"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "15.00",
            "https://inworld.ai/pricing",
            "On-demand: Realtime TTS 1.5 Mini $15 per 1M characters",
        ),
    ),
    (Benchmark.TTS, "minimax", "speech-2.8-hd"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "100.00",
            "https://platform.minimax.io/docs/guides/pricing-paygo",
            "Pay-as-you-go table: T2A speech-2.8-hd $100/M characters",
        ),
    ),
    (Benchmark.TTS, "minimax", "speech-2.8-turbo"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "60.00",
            "https://platform.minimax.io/docs/guides/pricing-paygo",
            "Pay-as-you-go table: T2A speech-2.8-turbo $60/M characters",
        ),
    ),
    (Benchmark.TTS, "rime", "coda"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "50.00",
            "https://www.rime.ai/pricing",
            'Coda "$0.05 / 1K characters" ×1000',
        ),
    ),
    (Benchmark.TTS, "rime", "arcana"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "40.00",
            "https://www.rime.ai/resources/introducing-new-pricing",
            'Starter plan: "Arcana $40 / million characters" (Rime pricing announcement; '
            "current /pricing page lists only Coda and Mist v3)",
        ),
    ),
    (Benchmark.TTS, "rime", "mistv3"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "30.00",
            "https://www.rime.ai/pricing",
            'Mist v3 "$0.03 / 1K characters" ×1000',
        ),
    ),
    (Benchmark.TTS, "lmnt", "blizzard"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "49.00",
            "https://www.lmnt.com/pricing",
            "Pro: $49/mo, 1.25M characters included, $0.045 per 1K after "
            "(plan-level pricing, no PAYG or per-model rates)",
            "LMNT Pro plan $49/mo includes 1.25M chars; 1M chars/month fits the "
            "quota: $49 / 1M chars",
        ),
    ),
    (Benchmark.TTS, "smallest", "lightning_v3.1_pro"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "19.50",
            "https://smallest.ai/pricing/models",
            "Lightning V3.1 Pro $0.195/10K characters ×100 (pay-as-you-go)",
        ),
    ),
    (Benchmark.TTS, "speechify", "simba-3.2"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "10.00",
            "https://speechify.ai/pricing",
            "Starter: $10/mo with 1M chars included, then $10/1M; rates uniform across "
            "Simba models",
            "Speechify Starter plan $10/mo includes 1M chars ($10/1M overage): $10 / 1M chars",
        ),
    ),
    (Benchmark.TTS, "speechify", "simba-3.0"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "10.00",
            "https://speechify.ai/pricing",
            "Starter: $10/mo with 1M chars included, then $10/1M; rates uniform across "
            "Simba models",
            "Speechify Starter plan $10/mo includes 1M chars ($10/1M overage): $10 / 1M chars",
        ),
    ),
    (Benchmark.TTS, "fishaudio", "s1"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "15.00",
            "https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits",
            "s1: $15.00 / M UTF-8 bytes (1 byte = 1 char for English)",
        ),
    ),
    (Benchmark.TTS, "fishaudio", "s2.1-pro"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "15.00",
            "https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits",
            "s2.1-pro: $15.00 / M UTF-8 bytes",
        ),
    ),
    (Benchmark.TTS, "fishaudio", "s2.1-pro-free"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "0.00",
            "https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits",
            "s2.1-pro-free: $0.00 / M UTF-8 bytes (genuinely listed at $0.00)",
        ),
    ),
    (Benchmark.TTS, "elevenlabs", "eleven_flash_v2_5"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "50.00",
            "https://elevenlabs.io/pricing/api",
            'Flash/Turbo TTS "$0.05 per 1K characters" ×1000; API billed in USD, not credits',
        ),
    ),
    (Benchmark.TTS, "elevenlabs", "eleven_multilingual_v2"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "100.00",
            "https://elevenlabs.io/pricing/api",
            'Multilingual v2/v3 TTS "$0.10 per 1K characters" ×1000',
        ),
    ),
    (Benchmark.TTS, "elevenlabs", "eleven_v3"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "100.00",
            "https://elevenlabs.io/pricing/api",
            "v3 priced identically to Multilingual v2 on the PAYG API page: $0.10/1k ×1000",
        ),
    ),
    (Benchmark.TTS, "cartesia", "sonic-3.5"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "39.20",
            "https://cartesia.ai/pricing",
            '"Startup $49 ... 1.25M TTS credits"; docs: standard TTS ≈ 1 credit/char',
            "Cartesia Startup plan $49/mo (1.25M credits, ~1 credit/char): $49 / 1.25M × 1M chars",
        ),
    ),
    (Benchmark.TTS, "deepgram", "aura-2-thalia-en"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "30.00",
            "https://deepgram.com/pricing",
            'Aura-2 TTS pay-as-you-go "$0.030/1k characters" ×1000',
        ),
    ),
    (Benchmark.TTS, "openai", "gpt-4o-mini-tts"): (
        Rate(
            BillingUnit.PER_1M_TOKENS_INPUT,
            "0.60",
            "https://developers.openai.com/api/docs/pricing",
            "gpt-4o-mini-tts text input tokens: $0.60 per 1M",
        ),
        Rate(
            BillingUnit.PER_1M_TOKENS_OUTPUT,
            "12.00",
            "https://developers.openai.com/api/docs/pricing",
            "gpt-4o-mini-tts audio output tokens: $12.00 per 1M",
        ),
    ),
    (Benchmark.TTS, "google", "chirp-3-hd"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "30.00",
            "https://cloud.google.com/text-to-speech/pricing",
            '"Chirp 3: HD voices (sku F977-2280-6F1B) US$0.00003 per character '
            '(US$30 per 1 million characters)"',
        ),
    ),
    (Benchmark.TTS, "hume", "octave-tts"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "70.00",
            "https://www.hume.ai/pricing",
            'Pro "$70/month ... 1,000,000 (~1,000 minutes)" included characters',
            "Hume Pro plan $70/mo includes exactly 1M characters: $70 / 1M chars",
        ),
    ),
    (Benchmark.TTS, "hume", "octave-2"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "70.00",
            "https://www.hume.ai/pricing",
            'characters billed identically for "Octave 1 / Octave 2" on the pricing page',
            "Hume Pro plan $70/mo includes exactly 1M characters: $70 / 1M chars",
        ),
    ),
    (Benchmark.TTS, "murf", "falcon-2"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "10.00",
            "https://help.murf.ai/murf-api-pay-as-you-go",
            '"Falcon model: $0.01/1000 characters" ×1000 (PAYG, $10 minimum purchase)',
        ),
    ),
    (Benchmark.TTS, "palabra", "palabra-tts-v1"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "30.00",
            "https://www.palabra.ai/pricing",
            'TTS pay-as-you-go "$0.03 / 1,000 CHARACTERS" ×1000',
        ),
    ),
    (Benchmark.TTS, "soniox", "tts-rt-v1"): (
        Rate(
            BillingUnit.PER_1M_TOKENS_INPUT,
            "4.00",
            "https://soniox.com/pricing",
            'real-time TTS "Input text tokens: $4.00 per 1M tokens"',
        ),
        Rate(
            BillingUnit.PER_1M_TOKENS_OUTPUT,
            "21.50",
            "https://soniox.com/pricing",
            '"Output audio tokens: $21.50 per 1M tokens" (~30k tokens ≈ 1h of speech)',
        ),
    ),
    (Benchmark.TTS, "gradium", "default"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "48.00",
            "https://gradium.ai/pricing",
            '"Text-to-Speech: 1 credit / character"; plans XS $13/225k, S $43/900k credits',
            "Gradium S plan $43/mo (900k credits, 1 credit/char) + 100k add-on "
            "credits at $5.00: $48 for 1M chars/month",
        ),
    ),
    (Benchmark.TTS, "xai", "grok-tts"): (
        Rate(
            BillingUnit.PER_1M_CHARS,
            "15.00",
            "https://docs.x.ai/docs/models",
            'Voice pricing table: "Text to Speech ... $15.00 / 1M chars"',
        ),
    ),
    # ------------------------------------------------------------------ S2S
    (Benchmark.S2S, "openai", "gpt-realtime"): (
        Rate(
            BillingUnit.PER_1M_TOKENS_INPUT,
            "32.00",
            "https://developers.openai.com/api/docs/pricing",
            "gpt-realtime audio input tokens: $32.00 per 1M (cached audio input $0.40)",
        ),
        Rate(
            BillingUnit.PER_1M_TOKENS_OUTPUT,
            "64.00",
            "https://developers.openai.com/api/docs/pricing",
            "gpt-realtime audio output tokens: $64.00 per 1M",
        ),
    ),
    (Benchmark.S2S, "google", "gemini-live"): (
        Rate(
            BillingUnit.PER_1M_TOKENS_INPUT,
            "3.00",
            "https://ai.google.dev/gemini-api/docs/pricing",
            "gemini-2.5-flash-native-audio-preview-12-2025 (Live API): "
            "audio input $3.00 per 1M tokens",
        ),
        Rate(
            BillingUnit.PER_1M_TOKENS_OUTPUT,
            "12.00",
            "https://ai.google.dev/gemini-api/docs/pricing",
            "gemini-2.5-flash-native-audio-preview-12-2025 (Live API): "
            "audio output $12.00 per 1M tokens",
        ),
    ),
    (Benchmark.S2S, "xai", "grok-voice-think-fast-1.0"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.05",
            "https://docs.x.ai/docs/models",
            '"grok-voice-think-fast-1.0 ... $0.05 / min ($3.00 / hr) audio"',
        ),
    ),
    (Benchmark.S2S, "xai", "grok-voice-think-fast-2.0"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.08",
            "https://docs.x.ai/docs/models",
            '"grok-voice-think-fast-2.0 ... $0.08 / min ($4.80 / hr) audio"',
        ),
    ),
}


# Rates for measurement instruments that are not benchmarked models themselves
# and so live outside the registry — the whisper-1 TTS judge. Same provenance
# rules; applied unconditionally by apply_ratesheet.
JUDGE_RATESHEET: dict[tuple[Benchmark, str, str], tuple[Rate, ...]] = {
    (Benchmark.TTS, "openai", "whisper-1-judge"): (
        Rate(
            BillingUnit.PER_MINUTE,
            "0.006",
            "https://developers.openai.com/api/docs/pricing",
            'Transcription table: "Whisper $0.006 / minute" (whisper-1 API model, '
            "used as the TTS WER judge)",
        ),
    ),
}


async def apply_ratesheet(store: PricingStore) -> tuple[int, int, list[str]]:
    """Upsert every ratesheet entry for the active roster, plus judge rates.

    Returns ``(inserted, unchanged, gaps)`` where *gaps* lists active models
    with no ratesheet entry.
    """
    inserted = 0
    unchanged = 0
    gaps: list[str] = []
    # One shared timestamp per invocation: a token-billed model's input+output
    # rows land as a single effective period, not two breakpoints ms apart.
    effective_at = datetime.now(tz=UTC)
    active = [
        m for m in MODEL_REGISTRY if m.status in (ModelStatus.ACTIVE, ModelStatus.EARLY_ACCESS)
    ]
    work: list[tuple[Benchmark, str, str, tuple[Rate, ...]]] = []
    for m in active:
        rates = RATESHEET.get((m.benchmark, m.provider, m.model))
        if not rates:
            gaps.append(f"{m.benchmark}:{m.provider}/{m.model}")
            continue
        work.append((m.benchmark, m.provider, m.model, rates))
    work.extend((b, p, mod, rates) for (b, p, mod), rates in JUDGE_RATESHEET.items())

    for benchmark, provider, model, rates in work:
        for r in rates:
            _, created = await store.upsert_rate(
                provider=provider,
                model=model,
                benchmark=benchmark,
                billing_unit=r.unit,
                rate_usd=Decimal(r.rate_usd),
                source_url=r.source_url,
                as_of=AS_OF,
                evidence=r.evidence,
                plan_assumption=r.plan_assumption,
                updated_by="human",
                effective_at=effective_at,
            )
            if created:
                inserted += 1
            else:
                unchanged += 1
    return inserted, unchanged, gaps


def _assert_local(database_url: str) -> None:
    """Refuse to seed against anything but a local DB (same guard as seed_arena)."""
    host = urlsplit(database_url).hostname
    if host not in _LOCAL_HOSTS:
        raise SystemExit(
            f"refusing to seed pricing: DB host {host!r} is not local "
            f"({sorted(_LOCAL_HOSTS)}). This script is dev-only."
        )


async def run_seed() -> None:
    settings = get_settings()
    _assert_local(settings.database_url)
    async with lifespan_pool(settings) as pool:
        store = PricingStore(pool)
        inserted, unchanged, gaps = await apply_ratesheet(store)
    print(f"pricing seed: {inserted} rates inserted, {unchanged} unchanged")
    if gaps:
        print(f"NO PUBLISHED PRICE ({len(gaps)} models — no row written, never guessed):")
        for gap in gaps:
            print(f"  - {gap}")


if __name__ == "__main__":
    asyncio.run(run_seed())
