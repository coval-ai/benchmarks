# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry of public pricing pages the price collector scrapes.

One entry per provider in ``registries/models.py``. ``None`` documents that
the provider has no public pricing page (dedicated infra, enterprise/contact
sales, console-only, or unreleased) — the collector skips it and coverage is
enforced by ``scripts/check_pricing_coverage.py`` instead. ``parse_hints``
feed the LLM extraction prompt: how the page states rates, which sections
matter, known traps.
"""

from __future__ import annotations

from pydantic import BaseModel


class PricingSource(BaseModel, frozen=True):
    """Where and how a provider publishes its list prices."""

    url: str
    notes: str = ""
    parse_hints: str = ""


PRICING_SOURCES: dict[str, PricingSource | None] = {
    "assemblyai": PricingSource(
        url="https://www.assemblyai.com/pricing",
        parse_hints=(
            "Streaming STT rates are per hour and billed on WebSocket session "
            "duration; Universal-Streaming and Universal-3.5 Pro Realtime rows."
        ),
    ),
    # Console-only pricing on international Model Studio; docs list models but no rates.
    "alibaba": None,
    "azure": PricingSource(
        url="https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/",
        parse_hints="Speech service pay-as-you-go table; realtime STT per audio hour.",
    ),
    # Dedicated inference we host ourselves — spend is instance-hours, not list rates.
    "baseten": None,
    "cartesia": PricingSource(
        url="https://cartesia.ai/pricing",
        notes="Credit plans only; effective rates assume the Startup plan.",
        parse_hints=(
            "Plans are credit-based: TTS ~1 credit/char, STT 3 credits/sec. Extract "
            "plan price + included credits, and any stated per-unit equivalents."
        ),
    ),
    # Enterprise/contact-sales; only a free trial is public.
    "deepdub": None,
    "deepgram": PricingSource(
        url="https://deepgram.com/pricing",
        parse_hints=(
            "Streaming STT per-minute rates (promo prices may show struck-through "
            "regular rates — extract the current charged rate); Nova-2 legacy rate "
            "lives in the FAQ; Aura-2 TTS is per 1k characters."
        ),
    ),
    "elevenlabs": PricingSource(
        url="https://elevenlabs.io/pricing/api",
        parse_hints=(
            "API pricing page billed in USD (not credits): TTS per 1k characters by "
            "model family (Flash/Turbo vs Multilingual/v3), Scribe STT per hour."
        ),
    ),
    "fishaudio": PricingSource(
        url="https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits",
        parse_hints="Per-model rates in USD per 1M UTF-8 bytes.",
    ),
    # Early access; no published pricing.
    "fluxions": None,
    "gladia": PricingSource(
        url="https://www.gladia.io/pricing",
        parse_hints="Starter PAYG real-time rate per hour; no per-model split.",
    ),
    "google": PricingSource(
        url="https://cloud.google.com/speech-to-text/pricing",
        notes="STT only; TTS rates live on cloud.google.com/text-to-speech/pricing.",
        parse_hints=(
            "Speech-to-Text V2 Recognition table: Standard tier per-minute rate "
            "(chirp models bill as Standard V2); first 500k min/month tier."
        ),
    ),
    "gradium": PricingSource(
        url="https://gradium.ai/pricing",
        notes="Credit plans only; effective rates carry plan assumptions.",
        parse_hints="Credits per second (STT) / per character (TTS) plus plan credit quotas.",
    ),
    "groq": PricingSource(
        url="https://groq.com/pricing",
        parse_hints="Audio models table; STT per hour of audio, TTS per 1M characters.",
    ),
    "hume": PricingSource(
        url="https://www.hume.ai/pricing",
        notes="Subscription plans; effective rates assume the Pro plan.",
        parse_hints="Plan price and included characters for Octave TTS.",
    ),
    "inworld": PricingSource(
        url="https://inworld.ai/pricing",
        parse_hints="On-demand per-1M-characters rates per TTS tier; STT per hour.",
    ),
    "lmnt": PricingSource(
        url="https://www.lmnt.com/pricing",
        notes="Plan-level pricing, no per-model rates.",
        parse_hints="Plan price, included characters, and per-1k overage rate.",
    ),
    "minimax": PricingSource(
        url="https://platform.minimax.io/docs/guides/pricing-paygo",
        parse_hints="Pay-as-you-go T2A table, USD per 1M characters per model.",
    ),
    "mistral": PricingSource(
        url="https://mistral.ai/pricing",
        notes=(
            "Voxtral transcription rates have historically lived in announcement "
            "posts rather than the main pricing page."
        ),
        parse_hints="Audio/transcription section; per-minute rates for Voxtral models.",
    ),
    "modulate": PricingSource(
        url="https://www.modulate.ai/api-pricing",
        parse_hints=("Per-hour streaming rates: Multilingual STT vs English Fast STT rows."),
    ),
    "murf": PricingSource(
        url="https://help.murf.ai/murf-api-pay-as-you-go",
        parse_hints="PAYG per-1k-character rates by model (Falcon vs Gen 2).",
    ),
    "openai": PricingSource(
        url="https://developers.openai.com/api/docs/pricing",
        parse_hints=(
            "Transcription table (audio-in/text-out token rates per 1M plus per-minute "
            "estimates), TTS token rates, and the Realtime models table "
            "(gpt-realtime-whisper is per minute; gpt-realtime bills audio tokens)."
        ),
    ),
    "palabra": PricingSource(
        url="https://www.palabra.ai/pricing",
        parse_hints="PAYG TTS per 1k characters; S2S per minute.",
    ),
    "revai": PricingSource(
        url="https://www.rev.ai/pricing",
        parse_hints="Reverb per-hour rate; streaming bills at the same model rate.",
    ),
    "rime": PricingSource(
        url="https://www.rime.ai/pricing",
        notes="Arcana rate last seen on rime.ai/resources/introducing-new-pricing.",
        parse_hints="Per-1k-character rates per model (Coda, Mist v3, Arcana).",
    ),
    "smallest": PricingSource(
        url="https://smallest.ai/pricing/models",
        parse_hints=(
            "Per-model PAYG rates: TTS per 10k characters, STT per minute "
            "(Pulse realtime vs batch)."
        ),
    ),
    "soniox": PricingSource(
        url="https://soniox.com/pricing",
        parse_hints=(
            "stt-rt per-hour rate (token billing underneath); realtime TTS input/output "
            "token rates per 1M."
        ),
    ),
    "speechify": PricingSource(
        url="https://speechify.ai/pricing",
        notes="Plan-level pricing, uniform across Simba models.",
        parse_hints="Plan price, included characters, per-1M overage.",
    ),
    "speechmatics": PricingSource(
        url="https://www.speechmatics.com/pricing",
        notes="Table is client-side rendered from a CSV the page loads.",
        parse_hints="Real-time Standard vs Enhanced per-hour rates (Pro plan column).",
    ),
    "together": PricingSource(
        url="https://www.together.ai/pricing",
        parse_hints=(
            "'Price per audio minute' table; prefer streaming variants "
            "(Whisper Large v3 Streaming, Nemotron ASR, Parakeet)."
        ),
    ),
    "xai": PricingSource(
        url="https://docs.x.ai/docs/models",
        parse_hints=(
            "Voice pricing table: STT per hour (REST vs streaming — use streaming), "
            "TTS per 1M chars, grok-voice per minute of audio."
        ),
    ),
}
