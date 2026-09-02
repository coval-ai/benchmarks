# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""The pricing log: pricing_rates, seeded from the packaged ratesheets.

Revision ID: 20260902_0025
Revises:     20260831_0024
Create Date: 2026-09-02

Rates move from the JSON ratesheets shipped in the package to this table, so a
price changes through the admin API instead of a deploy, and every price ever
served stays on record. Append-only: a row is a *recording* — on ``recorded_at``
someone said that from ``effective_from`` the model costs ``price_usd`` per
``unit``, or (both NULL) that no public rate is known from that day. Nothing is
ever updated or deleted; ``registries.pricing.resolve`` reads the log (latest
recording wins per effective date; the rate in force on a day is the winning
recording with the greatest effective date at or before it).

No FK to ``models``: like ``model_history``, the log must outlive anything it
references. No uniqueness on the rate itself, on purpose — recording a value
again after a correction is a real change, and the stores decide what counts as
a repeat.

The seed is the literal content of the ratesheets at this revision (the
``SEED_ROWS`` below, generated from ``registries/pricing/data/``), as
20260825_0021 did for the model registry: a migration's effect must not drift
with the code that later ships beside it. ``recorded_by_user_id`` names this
migration; the first admin recording is the first human row.

Notes
-----
DB roles (``runner``, ``api``) and GRANTs are managed by Terraform. Admin
writes happen in the API process, so the ``api`` role needs SELECT and INSERT
on ``pricing_rates``; ``coval-bench pricing sync`` runs wherever the runner
does, so the ``runner`` role needs the same. Nothing is ever updated or
deleted.
"""

from __future__ import annotations

from alembic import op

revision = "20260902_0025"
down_revision = "20260831_0024"
branch_labels = None
depends_on = None

_RECORDED_BY = f"migration:{revision}"

# (benchmark, provider, model, unit, price_usd, effective_from, source_url, notes)
SEED_ROWS: tuple[tuple[str, str, str, str, str, str, str, str | None], ...] = (
    (
        "STT",
        "assemblyai",
        "universal-3.5-pro",
        "per_hour",
        "0.45",
        "2026-08-24",
        "https://www.assemblyai.com/pricing",
        "AssemblyAI sells three separate APIs and prices Universal-3.5 Pro differently on each. This is the Realtime Speech-to-Text API rate, listed as 'Universal-3.5 Pro Realtime' at $0.45/hr — the endpoint we benchmark, over wss://streaming.assemblyai.com/v3/ws. The similarly named async model on the Pre-recorded API is $0.21/hr and the Sync API is $0.45/hr; neither is what we measure. Billed on session duration like the Universal-Streaming rate above.",
    ),
    (
        "STT",
        "assemblyai",
        "universal-streaming",
        "per_hour",
        "0.15",
        "2026-08-24",
        "https://www.assemblyai.com/pricing",
        "Pay-as-you-go rate for Universal-Streaming on the Realtime Speech-to-Text API (English and Multilingual both $0.15/hr). AssemblyAI bills streaming on session duration — WebSocket open time, not audio sent — and we stream each clip in real time and close, so a benchmarked session is one clip's audio plus a brief wait for the final transcript.",
    ),
    (
        "STT",
        "assemblyai",
        "universal-streaming-multilingual",
        "per_hour",
        "0.15",
        "2026-08-31",
        "https://www.assemblyai.com/pricing",
        "'Universal-Streaming Multilingual' row on the Realtime Speech-to-Text API, same $0.15/hr as the English tier. Billed on session duration like the rates above.",
    ),
    (
        "STT",
        "azure",
        "default",
        "per_hour",
        "1",
        "2026-08-31",
        "https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/",
        "Pay-as-you-go 'S1 Speech To Text' standard real-time meter, $1.00 per hour in East US (the pricing page renders after region selection; figure verified via the Azure retail prices API). Batch and commitment tiers are cheaper.",
    ),
    (
        "STT",
        "deepgram",
        "flux-general-en",
        "per_minute",
        "0.0065",
        "2026-08-24",
        "https://deepgram.com/pricing",
        "Pay As You Go streaming rate, listed as a limited-time promotional price; regular price $0.0077/min.",
    ),
    (
        "STT",
        "deepgram",
        "flux-general-multi",
        "per_minute",
        "0.0078",
        "2026-08-24",
        "https://deepgram.com/pricing",
        "Pay As You Go streaming rate.",
    ),
    (
        "STT",
        "deepgram",
        "nova-2",
        "per_hour",
        "0.35",
        "2026-08-31",
        "https://deepgram.com/pricing",
        "FAQ rate: 'Nova-2 streaming at $0.35/hour', kept unchanged for existing deployments; Nova-2 no longer appears in the main rate tables. Growth plan rates are about 12.5% lower.",
    ),
    (
        "STT",
        "deepgram",
        "nova-3",
        "per_minute",
        "0.0048",
        "2026-08-24",
        "https://deepgram.com/pricing",
        "Pay As You Go monolingual (English) streaming rate, listed as a limited-time promotional price; regular price $0.0077/min. Multilingual streaming is $0.0058/min (regular $0.0092).",
    ),
    (
        "STT",
        "elevenlabs",
        "scribe_v2_realtime",
        "per_hour",
        "0.39",
        "2026-08-24",
        "https://elevenlabs.io/pricing/api",
        "Pay-as-you-go API rate for Scribe v2 Realtime; subscription tiers change only the included hours, not the rate.",
    ),
    (
        "STT",
        "gladia",
        "solaria-1",
        "per_hour",
        "0.75",
        "2026-08-24",
        "https://www.gladia.io/pricing",
        "Starter pay-as-you-go real-time rate; Gladia prices real-time generically and Solaria is its real-time model.",
    ),
    (
        "STT",
        "google",
        "chirp_2",
        "per_minute",
        "0.016",
        "2026-08-24",
        "https://cloud.google.com/speech-to-text/pricing",
        "Speech-to-Text V2 Recognition standard rate, first volume tier (0-500k min/month); V2 prices all standard models, Chirp included, at this one rate. Volume tiers drop to $0.004/min above 2M.",
    ),
    (
        "STT",
        "google",
        "chirp_3",
        "per_minute",
        "0.016",
        "2026-08-24",
        "https://cloud.google.com/speech-to-text/pricing",
        "Speech-to-Text V2 Recognition standard rate, first volume tier (0-500k min/month); same single V2 SKU as chirp_2.",
    ),
    (
        "STT",
        "gradium",
        "default",
        "per_second_audio_in",
        "0.000207",
        "2026-08-24",
        "https://gradium.ai/pricing",
        "Entry (XS) plan add-on credit rate — the marginal price of usage: 3 credits per second of audio at $6.9 per 100k credits, exact conversion. Gradium has no standalone pay-as-you-go tier; larger plans publish lower add-on rates, and usage inside the XS plan's own 225k-credit allowance works out to about $0.000173 per second.",
    ),
    (
        "STT",
        "inworld",
        "inworld-stt-1",
        "per_hour",
        "0.15",
        "2026-08-24",
        "https://inworld.ai/pricing",
        "On-Demand pay-as-you-go base rate; paid subscription tiers discount to $0.10/hr.",
    ),
    (
        "STT",
        "mistral",
        "voxtral-mini-transcribe-realtime-2602",
        "per_minute",
        "0.006",
        "2026-08-24",
        "https://mistral.ai/pricing/api",
        "La Plateforme pay-as-you-go audio-input rate for Voxtral Mini Transcribe Realtime; the batch Transcribe tier is $0.003/min.",
    ),
    (
        "STT",
        "modulate",
        "english-fast-transcription-streaming",
        "per_hour",
        "0.05",
        "2026-08-31",
        "https://www.modulate.ai/api-pricing",
        "'English Fast Speech-to-Text' streaming (real-time WebSocket) rate; the batch REST tier is $0.025/hr.",
    ),
    (
        "STT",
        "modulate",
        "multilingual-transcription-streaming",
        "per_hour",
        "0.06",
        "2026-08-31",
        "https://www.modulate.ai/api-pricing",
        "'Multilingual Speech-to-Text' streaming (real-time WebSocket) rate; batch is $0.03/hr, and emotion, accent, and PII tagging bill as separate add-ons. Modulate's page prices these two products only — the velma-2 model ids have no printed row and stay unpriced.",
    ),
    (
        "STT",
        "openai",
        "gpt-4o-mini-transcribe",
        "per_minute",
        "0.003",
        "2026-08-24",
        "https://developers.openai.com/api/docs/pricing",
        "OpenAI's published estimated per-minute cost; underlying token rates are $1.25/1M input tokens and $5.00/1M output tokens.",
    ),
    (
        "STT",
        "openai",
        "gpt-4o-transcribe",
        "per_minute",
        "0.006",
        "2026-08-24",
        "https://developers.openai.com/api/docs/pricing",
        "OpenAI's published estimated per-minute cost; underlying token rates are $2.50/1M input tokens and $10.00/1M output tokens.",
    ),
    (
        "STT",
        "openai",
        "gpt-realtime-whisper",
        "per_minute",
        "0.017",
        "2026-08-24",
        "https://developers.openai.com/api/docs/pricing",
        "Published per-minute live-transcription rate; OpenAI lists no token pricing for this model.",
    ),
    (
        "STT",
        "revai",
        "reverb",
        "per_hour",
        "0.20",
        "2026-08-31",
        "https://www.rev.ai/pricing",
        "'Reverb Transcription $0.20 per hour' — Rev AI publishes one rate per product with no streaming/async split; usage is rounded up to the nearest second with a 15-second minimum. Reverb Turbo is $0.10/hr and foreign-language transcription $0.30/hr.",
    ),
    (
        "STT",
        "smallest",
        "pulse",
        "per_minute",
        "0.008",
        "2026-08-24",
        "https://docs.smallest.ai/models/model-cards/speech-to-text/pulse",
        "Standard Plan realtime rate from the official Pulse model card; batch is $0.005/min. Marketing pages print only approximate figures.",
    ),
    (
        "STT",
        "soniox",
        "stt-rt-v5",
        "per_hour",
        "0.12",
        "2026-08-24",
        "https://soniox.com/pricing",
        "Soniox's page-printed per-hour rate for real-time streaming, which is the all-in figure: canonical billing is token-based, and the page's own conversions add up to it — ~30,000 input audio tokens per hour at $2.00 per 1M ($0.06) plus ~15,000 output text tokens per hour at $4.00 per 1M ($0.06).",
    ),
    (
        "STT",
        "speechmatics",
        "default",
        "per_hour",
        "0.24",
        "2026-08-24",
        "https://www.speechmatics.com/pricing",
        "Pro pay-as-you-go rate for real-time transcription, Standard operating point.",
    ),
    (
        "STT",
        "speechmatics",
        "enhanced",
        "per_hour",
        "0.43",
        "2026-08-24",
        "https://www.speechmatics.com/pricing",
        "Pro pay-as-you-go rate for real-time transcription, Enhanced operating point.",
    ),
    (
        "STT",
        "together",
        "nemotron-3-asr-streaming-0.6b",
        "per_minute",
        "0.0015",
        "2026-08-31",
        "https://www.together.ai/pricing",
        "'NVIDIA Nemotron 3 ASR Streaming 0.6B' row in the Transcribe per-audio-minute table; no separate batch discount is listed for it. Distinct from the newer Nemotron 3.5 model above at $0.0045/min.",
    ),
    (
        "STT",
        "together",
        "nemotron-3.5-asr-streaming-0.6b",
        "per_minute",
        "0.0045",
        "2026-08-24",
        "https://www.together.ai/models/nvidia-nemotron-35-asr",
        "Serverless pay-as-you-go per-audio-minute rate; the model page prints this price against the API string nvidia/nemotron-3.5-asr-streaming-0.6b (the pricing page's 'NVIDIA Nemotron 3.5 ASR' row). Not the older 'Nemotron 3 ASR Streaming 0.6B' row at $0.0015/min.",
    ),
    (
        "STT",
        "together",
        "parakeet-tdt-0.6b-v3",
        "per_minute",
        "0.0015",
        "2026-08-24",
        "https://www.together.ai/pricing",
        "Serverless per-audio-minute rate for NVIDIA Parakeet TDT 0.6B v3. Together's per-audio-minute table is headed 'Batch API price' and carries no realtime row for this model, unlike Whisper Large v3 which has a separate (Streaming) row — so this is the only per-minute figure published for it, and it may understate the realtime endpoint we benchmark. The 'Parakeet TDT 0.6B V3 Realtime $0.0035' row sits in the per-1M-characters table beside the TTS models, so it is not a per-minute audio rate.",
    ),
    (
        "STT",
        "together",
        "whisper-large-v3",
        "per_minute",
        "0.0035",
        "2026-08-24",
        "https://www.together.ai/pricing",
        "Whisper Large v3 (Streaming) rate — we benchmark over Together's realtime API; the non-streaming Transcribe tier is $0.0015/min.",
    ),
    (
        "STT",
        "xai",
        "grok-stt",
        "per_hour",
        "0.20",
        "2026-08-24",
        "https://docs.x.ai/developers/pricing",
        "Streaming speech-to-text rate — the endpoint we benchmark; the REST tier is $0.10/hr. Published as a generic Speech to Text row, not by model id.",
    ),
    (
        "TTS",
        "alibaba",
        "qwen3-tts-flash-realtime",
        "per_1m_chars",
        "13",
        "2026-09-01",
        "https://www.alibabacloud.com/help/en/model-studio/model-pricing",
        "Qwen3-TTS-Flash-Realtime table, Singapore region ('International' deployment scope — the dashscope-intl endpoint we benchmark): $0.13 per 10,000 input text characters, output audio free. The Chinese-mainland (Beijing) table lists $0.143353 per 10,000.",
    ),
    (
        "TTS",
        "azure",
        "dragon-hd-latest",
        "per_1m_chars",
        "22",
        "2026-08-10",
        "https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/",
        "Pay-as-you-go Neural HD rate in East US; HD voices are region-limited.",
    ),
    (
        "TTS",
        "azure",
        "neural",
        "per_1m_chars",
        "15",
        "2026-08-10",
        "https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/",
        "Pay-as-you-go standard neural voice rate (US regions); commitment tiers are cheaper.",
    ),
    (
        "TTS",
        "deepgram",
        "aura-2-thalia-en",
        "per_1k_chars",
        "0.030",
        "2026-08-10",
        "https://deepgram.com/pricing",
        "Aura-2 Pay As You Go rate.",
    ),
    (
        "TTS",
        "elevenlabs",
        "eleven_flash_v2_5",
        "per_1k_chars",
        "0.05",
        "2026-08-10",
        "https://elevenlabs.io/pricing/api",
        "API pricing 'Flash / Turbo' row; same rate across tiers.",
    ),
    (
        "TTS",
        "elevenlabs",
        "eleven_multilingual_v2",
        "per_1k_chars",
        "0.10",
        "2026-08-31",
        "https://elevenlabs.io/pricing/api",
        "API pricing 'v2 Multilingual' row. Same rate across tiers.",
    ),
    (
        "TTS",
        "elevenlabs",
        "eleven_turbo_v2_5",
        "per_1k_chars",
        "0.05",
        "2026-08-31",
        "https://elevenlabs.io/pricing/api",
        "API pricing 'Flash / Turbo' row; the page's tooltip states the row covers Flash/Turbo V2 and V2.5. Same rate across tiers.",
    ),
    (
        "TTS",
        "elevenlabs",
        "eleven_v3",
        "per_1k_chars",
        "0.10",
        "2026-08-31",
        "https://elevenlabs.io/pricing/api",
        "API pricing 'v3' row — twice the v3 Conversational rate below. Same rate across tiers.",
    ),
    (
        "TTS",
        "elevenlabs",
        "eleven_v3_conversational",
        "per_1k_chars",
        "0.05",
        "2026-08-28",
        "https://elevenlabs.io/pricing/api",
        "API pricing 'v3 Conversational' row — a distinct model from plain v3, which is $0.10 per 1K characters. Same rate across tiers.",
    ),
    (
        "TTS",
        "fishaudio",
        "s1",
        "per_1m_chars",
        "15",
        "2026-08-31",
        "https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits",
        "Printed as '$15.00 / M UTF-8 bytes' — one byte per character for the English text we benchmark. s2.1-pro-free is listed at $0.00 and stays unpriced here (the registry cannot carry a zero rate).",
    ),
    (
        "TTS",
        "fishaudio",
        "s2.1-pro",
        "per_1m_chars",
        "15",
        "2026-08-31",
        "https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits",
        "Printed as '$15.00 / M UTF-8 bytes', same as s1; billed pay-as-you-go from the first call.",
    ),
    (
        "TTS",
        "fluxions",
        "vui",
        "per_1m_chars",
        "10",
        "2026-08-10",
        "https://fluxions.ai/pricing",
        "Pay-as-you-go rate for expressive VUI text-to-speech.",
    ),
    (
        "TTS",
        "google",
        "chirp-3-hd",
        "per_1m_chars",
        "30",
        "2026-08-10",
        "https://cloud.google.com/text-to-speech/pricing",
        "Pay-as-you-go rate for Chirp 3: HD voices, billed per character sent for synthesis.",
    ),
    (
        "TTS",
        "gradium",
        "default",
        "per_1m_chars",
        "69",
        "2026-08-10",
        "https://gradium.ai/pricing",
        "Entry (XS) plan add-on credit rate — the marginal price of usage: $6.9 per 100k credits at 1 credit per character. Gradium has no standalone pay-as-you-go tier; larger plans publish lower add-on rates, and usage inside the XS plan's own 225k-credit allowance works out to about $57.78 per 1M.",
    ),
    (
        "TTS",
        "gradium",
        "gradium-tts-beta",
        "per_1m_chars",
        "69",
        "2026-08-31",
        "https://gradium.ai/pricing",
        "Gradium prices TTS generically — '1 credit / character' with no model named on the page — so the new default model bills like the entry (XS) plan's marginal add-on rate recorded for gradium/default: $6.9 per 100k add-on credits = $69/1M characters. Larger plans publish $50/$40/$38.",
    ),
    (
        "TTS",
        "hume",
        "octave-2",
        "per_1k_chars",
        "0.15",
        "2026-08-31",
        "https://www.hume.ai/pricing",
        "Same Creator-plan overage rate as octave-tts above; the pricing page covers 'Octave 1 | Octave 2' with one set of figures.",
    ),
    (
        "TTS",
        "hume",
        "octave-tts",
        "per_1k_chars",
        "0.15",
        "2026-08-31",
        "https://www.hume.ai/pricing",
        "Creator plan ($7/mo) 'Additional characters cost' overage rate — the entry tier that prints one; Hume publishes no pay-as-you-go tier, and larger plans overage at $0.12/$0.10/$0.05 per 1K. Octave 1 and Octave 2 are priced jointly.",
    ),
    (
        "TTS",
        "inworld",
        "inworld-tts-2",
        "per_1m_chars",
        "25",
        "2026-08-10",
        "https://inworld.ai/pricing",
        "On-demand (pay-as-you-go) rate; subscription plans publish lower rates.",
    ),
    (
        "TTS",
        "inworld",
        "inworld-tts-2-flash",
        "per_1m_chars",
        "15",
        "2026-08-24",
        "https://inworld.ai/pricing",
        "On-demand (pay-as-you-go) rate; subscription plans publish lower rates. TTS 1.5 Max/Mini are no longer published on the page, so the retired models carry no rate.",
    ),
    (
        "TTS",
        "lmnt",
        "blizzard",
        "per_1k_chars",
        "0.05",
        "2026-08-10",
        "https://www.lmnt.com/pricing",
        "Overage rate past the entry Indie plan's included characters; plan-based, not per-model.",
    ),
    (
        "TTS",
        "minimax",
        "speech-2.8-hd",
        "per_1m_chars",
        "100",
        "2026-08-10",
        "https://platform.minimax.io/docs/guides/pricing-paygo.md",
        "International-platform pay-as-you-go rate.",
    ),
    (
        "TTS",
        "minimax",
        "speech-2.8-turbo",
        "per_1m_chars",
        "60",
        "2026-08-10",
        "https://platform.minimax.io/docs/guides/pricing-paygo.md",
        "International-platform pay-as-you-go rate.",
    ),
    (
        "TTS",
        "murf",
        "falcon-2",
        "per_1k_chars",
        "0.01",
        "2026-08-31",
        "https://murf.ai/pricing",
        "API tab of the pricing page, the realtime Text to Speech row ('Optimized for conversational AI and real-time voice agents with time-to-first-audio under 130ms', i.e. Falcon): $0.01 / 1000 characters. The $0.03 row beside it is the studio synthesis model, not this one. The page renders client-side; the old murf.ai/api/pricing route 404s since early Sep 2026.",
    ),
    (
        "TTS",
        "openai",
        "tts-1",
        "per_1m_chars",
        "15",
        "2026-08-31",
        "https://developers.openai.com/api/docs/pricing",
        "Printed as '$15.00 / 1M characters' in the audio-generation pricing table (the row is collapsed behind the table's expander). gpt-4o-mini-tts and the realtime models bill in tokens only and stay unpriced here.",
    ),
    (
        "TTS",
        "openai",
        "tts-1-hd",
        "per_1m_chars",
        "30",
        "2026-08-31",
        "https://developers.openai.com/api/docs/pricing",
        "Printed as '$30.00 / 1M characters' in the audio-generation pricing table (collapsed row), twice the tts-1 rate.",
    ),
    (
        "TTS",
        "palabra",
        "palabra-tts-v1",
        "per_1k_chars",
        "0.03",
        "2026-08-10",
        "https://www.palabra.ai/pricing",
        "TTS API pay-as-you-go rate.",
    ),
    (
        "TTS",
        "rime",
        "coda",
        "per_1k_chars",
        "0.05",
        "2026-08-10",
        "https://www.rime.ai/pricing",
        "Pay-as-you-go Starter rate. Arcana has no published rate and is deliberately absent.",
    ),
    (
        "TTS",
        "rime",
        "mistv3",
        "per_1k_chars",
        "0.03",
        "2026-08-10",
        "https://www.rime.ai/pricing",
        "Pay-as-you-go Starter rate.",
    ),
    (
        "TTS",
        "speechify",
        "simba-3.0",
        "per_1m_chars",
        "10",
        "2026-08-10",
        "https://speechify.ai/pricing",
        "Starter plan's rate past its 1M included characters — the entry published rate; Pro is $8 and Scale $6 per 1M. Billed per character at the same rate for every Simba model. The per-minute figure Speechify also publishes prices their bundled voice-agent product, not the TTS API.",
    ),
    (
        "TTS",
        "speechify",
        "simba-3.2",
        "per_1m_chars",
        "10",
        "2026-08-10",
        "https://speechify.ai/pricing",
        "Starter plan's rate past its 1M included characters — the entry published rate; Pro is $8 and Scale $6 per 1M. Billed per character at the same rate for every Simba model. The per-minute figure Speechify also publishes prices their bundled voice-agent product, not the TTS API.",
    ),
    (
        "TTS",
        "xai",
        "grok-tts",
        "per_1m_chars",
        "15",
        "2026-08-10",
        "https://docs.x.ai/developers/pricing",
        "Voice pricing 'Text to Speech' rate; xAI prices the service, not model variants.",
    ),
)


def upgrade() -> None:
    """Create the pricing log and seed it with the packaged rates."""
    op.execute(
        """
        CREATE TABLE benchmarks_v2.pricing_rates (
            id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            benchmark           TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S')),
            provider            TEXT NOT NULL CHECK (provider <> ''),
            model               TEXT NOT NULL CHECK (model <> ''),
            -- Both set: a published rate in the provider's native unit. Both NULL:
            -- no public rate is known from effective_from (a delisting).
            unit                TEXT CHECK (unit IS NULL OR unit <> ''),
            -- NUMERIC keeps the quoted scale: 0.20 stays 0.20, never 0.2 — the
            -- same exact-decimal contract /v1/pricing serves.
            price_usd           NUMERIC CHECK (price_usd IS NULL OR price_usd > 0),
            effective_from      DATE NOT NULL,
            source_url          TEXT CHECK (source_url IS NULL OR source_url <> ''),
            notes               TEXT CHECK (notes IS NULL OR notes <> ''),
            recorded_by_user_id TEXT NOT NULL CHECK (recorded_by_user_id <> ''),   -- Clerk sub, or migration:/pricing-sync
            recorded_by_email   TEXT CHECK (recorded_by_email IS NULL OR recorded_by_email <> ''), -- label at write time; scrubbable
            recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK ((unit IS NULL) = (price_usd IS NULL)),
            CHECK (price_usd IS NULL OR source_url IS NOT NULL)
        );
        CREATE INDEX pricing_rates_key_effective
            ON benchmarks_v2.pricing_rates (benchmark, provider, model, effective_from DESC, recorded_at DESC);
        """
    )
    bind = op.get_bind()
    for row in SEED_ROWS:
        # exec_driver_sql with parameters, not op.execute with literals: URLs and
        # notes carry colons and percent signs that text() and format() would eat.
        bind.exec_driver_sql(
            """
            INSERT INTO benchmarks_v2.pricing_rates
                (benchmark, provider, model, unit, price_usd, effective_from, source_url, notes,
                 recorded_by_user_id)
            VALUES (%s, %s, %s, %s, %s::numeric, %s::date, %s, %s, %s)
            """,
            (*row, _RECORDED_BY),
        )


def downgrade() -> None:
    """Drop the pricing log, admin recordings included — they have no other home."""
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.pricing_rates;")
