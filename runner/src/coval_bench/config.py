# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Pydantic Settings for the coval-bench runner and API.

Every other module that needs configuration imports from here:

    from coval_bench.config import Settings, get_settings
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Reserved: the aggregation layer materializes pooled rows under this sentinel.
DATASET_ALL = "__all__"


class Settings(BaseSettings):
    """Application settings, populated from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database ---
    # `str`, not `PostgresDsn`: in prod we use the Cloud SQL Auth Proxy Unix
    # socket form `postgresql://user:pw@/db?host=/cloudsql/<conn-name>`, which
    # Pydantic's `PostgresDsn` rejects (empty host). psycopg / SQLAlchemy
    # validate the URL at connect time, so the Pydantic-side check would only
    # block the legitimate prod form without catching anything new.
    #
    # Default placeholder lets provider-only CLIs (e.g. ``coval-bench tts-smoke``) run
    # without DATABASE_URL. Set DATABASE_URL for ``run``, migrate, API, and Docker Compose.
    database_url: str = Field(
        default="postgresql://unused:unused@127.0.0.1:5432/unused",
    )

    # --- Dataset ---
    dataset_bucket: str = "coval-benchmarks-datasets"
    dataset_id: str = "stt-v1"

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id_not_reserved(cls, value: str) -> str:
        if value == DATASET_ALL:
            raise ValueError(f"dataset_id {DATASET_ALL!r} is reserved for pooled aggregates")
        return value

    # Items drawn at random per run from each manifest, shared across all
    # models for parity. Set >= manifest size to run everything.
    dataset_sample_size: int = 10

    # --- Runner ---
    runner_sha: str = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Scheduler period in seconds. The runner floors its start time to this grid
    # to compute each run's scheduled_at. MUST stay in sync with the Cloud
    # Scheduler cron cadence (*/30 -> 1800, */15 -> 900); set via the
    # SCHEDULE_PERIOD_SECONDS env var, owned by the infra repo.
    schedule_period_seconds: int = Field(default=1800, gt=0)

    # --- Provider API keys (all optional; loaded from Secret Manager at runtime) ---
    openai_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None
    cartesia_api_key: SecretStr | None = None
    deepgram_api_key: SecretStr | None = None
    assemblyai_api_key: SecretStr | None = None
    speechmatics_api_key: SecretStr | None = None
    hume_api_key: SecretStr | None = None
    rime_api_key: SecretStr | None = None
    gladia_api_key: SecretStr | None = None
    gradium_api_key: SecretStr | None = None
    gradium_tts_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    xai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    smallest_api_key: SecretStr | None = None
    inworld_api_key: SecretStr | None = None
    soniox_api_key: SecretStr | None = None
    revai_api_key: SecretStr | None = None
    baseten_api_key: SecretStr | None = None
    together_api_key: SecretStr | None = None
    fishaudio_api_key: SecretStr | None = None
    azure_api_key: SecretStr | None = None
    alibaba_api_key: SecretStr | None = None
    minimax_api_key: SecretStr | None = None
    palabra_api_key: SecretStr | None = None

    # Azure region hosting the Speech resource (e.g. "eastus"). Determines the
    # region-scoped WebSocket host; required only when the Azure STT provider runs.
    azure_region: str | None = None

    # Baseten dedicated-endpoint WebSocket URLs. The hostnames embed private,
    # pre-launch model ids, so they live in config (``.env`` locally, Secret
    # Manager in prod) rather than hardcoded in the provider modules.
    baseten_whisper_url: str | None = None  # STT (Whisper Large V3)
    baseten_qwen_url: str | None = None  # TTS (Qwen3-TTS)

    alibaba_tts_url: str | None = None

    # Path to a Google service-account JSON file mounted as a Secret-as-volume.
    google_application_credentials: Path | None = None

    # GCP project ID hosting the Google STT v2 recognizer. Required only when
    # the Google STT provider is enabled (optional `google-stt` extra).
    google_project_id: str | None = None

    # --- Coval API (S2S fetch job) ---
    # X-API-Key for the Coval API. SecretStr so it never lands in a log.
    coval_api_key: SecretStr | None = None
    coval_api_base: str = "https://api.coval.dev/v1"
    # The S2S latency metric id + per-provider Coval agent ids (opaque, not secret).
    coval_s2s_latency_metric_id: str | None = None
    coval_s2s_openai_agent_id: str | None = None
    coval_s2s_gemini_agent_id: str | None = None
    coval_s2s_xai_agent_id: str | None = None
    # Fetch grid, in seconds; kept in sync with the s2s-fetch-trigger cron in
    # benchmark-infra (override via S2S_FETCH_PERIOD_SECONDS). Default = 3h.
    s2s_fetch_period_seconds: int = Field(default=10_800, gt=0)
    # Staleness threshold = fetch period + this grace.
    s2s_stale_grace_seconds: int = Field(default=0, ge=0)

    # --- Analytics ---
    posthog_project_token: str = ""
    posthog_host: str = "https://us.i.posthog.com"
    posthog_disabled: bool = False

    # --- API ---
    cors_origins: list[str] = [
        "https://benchmarks.coval.ai",
        "https://benchmarks-covalai.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ]
    # Matches Vercel preview deploys for the covalai/benchmarks project:
    # branch URLs (`benchmarks-git-<branch>-covalai.vercel.app`) and
    # per-deployment hash URLs (`benchmarks-<hash>-covalai.vercel.app`).
    # The canonical project URL is in cors_origins above; this regex is for
    # ephemeral preview deploys only.
    cors_origin_regex: str | None = r"^https://benchmarks-[a-z0-9-]+-covalai\.vercel\.app$"
    rate_limit_per_minute: int = 60
    # Benchmarking-team key (X-Internal-Key header): unlocks EARLY_ACCESS
    # models on the data endpoints. Unset means no request is internal.
    internal_api_key: SecretStr | None = None
    # Env-defined stealth models (JSON: alias -> real upstream); shape and
    # semantics in coval_bench.registries.stealth.
    stealth_models: SecretStr | None = None

    # --- Arena ---
    arena_labeler_key: SecretStr | None = None
    arena_audio_dir: Path = Path("arena-audio")
    arena_audio_base_url: str = ""
    arena_gcs_bucket: str = ""
    # Must match the GCS bucket's object-deletion lifecycle (set in benchmark-infra).
    arena_clip_retention_days: int = 30
    arena_daily_battle_cap: int = 500


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-cached Settings instance."""
    return Settings()
