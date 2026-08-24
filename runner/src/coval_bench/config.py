# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Pydantic Settings for the coval-bench runner and API.

Every other module that needs configuration imports from here:

    from coval_bench.config import Settings, get_settings
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Literal, get_args

import structlog
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Reserved: the aggregation layer materializes pooled rows under this sentinel.
DATASET_ALL = "__all__"

# Terraform's Secret Manager stub; a mount that was never rotated delivers it verbatim.
SECRET_PLACEHOLDER = "PLACEHOLDER_REPLACE_VIA_GCLOUD"  # noqa: S105 — a stub, not a credential


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
    # Private bucket for additive normalized observation artifacts. The rollout
    # is deliberately fail-closed: enabling writes without a destination is an
    # invalid deployment rather than a silent partial capture.
    benchmark_artifact_bucket: str = ""
    normalized_dual_write_enabled: bool = False

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id_not_reserved(cls, value: str) -> str:
        if value == DATASET_ALL:
            raise ValueError(f"dataset_id {DATASET_ALL!r} is reserved for pooled aggregates")
        return value

    @model_validator(mode="after")
    def _normalized_dual_write_requires_bucket(self) -> Settings:
        if self.normalized_dual_write_enabled and not self.benchmark_artifact_bucket:
            raise ValueError(
                "benchmark_artifact_bucket is required when normalized dual write is enabled"
            )
        return self

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
    reson8_api_key: SecretStr | None = None
    revai_api_key: SecretStr | None = None
    baseten_api_key: SecretStr | None = None
    together_api_key: SecretStr | None = None
    fishaudio_api_key: SecretStr | None = None
    azure_api_key: SecretStr | None = None
    alibaba_api_key: SecretStr | None = None
    minimax_api_key: SecretStr | None = None
    palabra_api_key: SecretStr | None = None
    lmnt_api_key: SecretStr | None = None
    murfai_api_key: SecretStr | None = None
    hakimai_api_key: SecretStr | None = None
    modulate_api_key: SecretStr | None = None
    speechify_api_key: SecretStr | None = None
    fluxions_api_key: SecretStr | None = None
    deepdub_api_key: SecretStr | None = None

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
    coval_s2s_xai_think_fast_2_agent_id: str | None = None
    coval_s2s_gray_agent_id: str | None = None
    coval_s2s_red_agent_id: str | None = None
    # The S2S instruction-adherence metric id (opaque, not secret). Optional: the
    # fetch pulls its per-conversation scores only when set, so latency still
    # ingests without it.
    coval_s2s_instruction_metric_id: str | None = None
    # The S2S interruption-rate metric id (opaque, not secret). Optional.
    coval_s2s_interruption_metric_id: str | None = None
    # Restrict the fetch to one Coval test set (the multi-turn set). Without it,
    # other sims on the same agents (e.g. the single-turn set) would be ingested
    # and pooled into the same provider. Opaque id, not secret.
    coval_s2s_test_set_id: str | None = None
    # Test set for agents evaluated on the happy-path scenarios instead of the
    # shared one above. Opaque id, not secret.
    coval_s2s_happypath_test_set_id: str | None = None
    # The caller persona whose audio carries background noise; its runs land
    # under their own dataset instead of pooling into the clean numbers. Unset
    # means every persona is clean. Superseded by coval_s2s_condition_personas.
    coval_s2s_noisy_persona_id: str | None = None
    # Caller persona id -> condition name, e.g. {"<id>": "noisy", "<id>": "clean"}.
    # Exhaustive: a persona absent from this map faults its provider rather than
    # counting as clean, which would be invisible in the data and the logs.
    coval_s2s_condition_personas: dict[str, str] = Field(default_factory=dict)
    # Fetch grid, in seconds; kept in sync with the s2s-fetch-trigger cron in
    # benchmark-infra (override via S2S_FETCH_PERIOD_SECONDS). Default = 3h.
    s2s_fetch_period_seconds: int = Field(default=10_800, gt=0)
    # Staleness threshold = fetch period + this grace.
    s2s_stale_grace_seconds: int = Field(default=0, ge=0)
    # Public bucket for per-tick sample recordings; empty disables sampling.
    s2s_samples_bucket: str = ""

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
    # Clerk instance whose JWKS verifies provider-org session tokens.
    # Unset means no bearer token unlocks anything.
    clerk_issuer: str | None = None
    # Allowed azp claim values; bearer tokens are rejected while empty.
    clerk_authorized_parties: list[str] = []
    # The coval staff org id: a token with this org active sees every embargoed
    # model. A clerk_org_providers or clerk_org_exclusive entry for this org takes
    # precedence and narrows it like any other org. Unset means no org gets the
    # staff view.
    clerk_coval_org: str | None = None
    # Clerk org id -> what it unlocks, as a JSON object. A value is a provider or a
    # list of providers and provider/model pairs: {"org_abc": "deepgram"},
    # {"org_def": ["colors/gray", "colors/red"]}. Keyed by the immutable org id, not
    # the slug, which org admins can rename. An org missing from the map unlocks
    # nothing.
    clerk_org_providers: str | None = None
    # Clerk org id -> the ONLY models it may see, as a JSON object:
    # {"org_abc": ["colors"]} or {"org_abc": ["colors/gray"]}. Same entry grammar as
    # clerk_org_providers, but exclusive: everything else, public models included, is
    # hidden from that org. Overrides clerk_org_providers for the same org.
    clerk_org_exclusive: str | None = None

    @field_validator("clerk_org_exclusive")
    @classmethod
    def _exclusive_map_is_usable(cls, value: str | None) -> str | None:
        """Refuse to start on an unusable exclusive map.

        The additive maps fall back to the public view when malformed, which is safe
        because they only ever widen. This one only ever narrows, so a broken blob
        must not boot into ordinary visibility for the orgs it was meant to restrict.
        """
        if value is None:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("clerk_org_exclusive is not valid JSON") from exc
        if not isinstance(parsed, dict) or not all(
            isinstance(org_id, str)
            and org_id
            and (
                (isinstance(entries, str) and entries)
                or (
                    isinstance(entries, list)
                    and all(isinstance(entry, str) and entry for entry in entries)
                )
            )
            for org_id, entries in parsed.items()
        ):
            raise ValueError(
                'clerk_org_exclusive must be {"org_id": "provider" | ["provider/model", ...]}'
            )
        return value

    # --- Arena ---
    arena_labeler_key: SecretStr | None = None
    arena_audio_dir: Path = Path("arena-audio")
    arena_audio_base_url: str = ""
    arena_gcs_bucket: str = ""
    # Must match the GCS bucket's object-deletion lifecycle (set in benchmark-infra).
    arena_clip_retention_days: int = 30
    arena_daily_battle_cap: int = 500
    # When the moderator cannot be reached, reject rather than synthesize: the local PII
    # check is not a content-safety fallback, and the arena publishes its audio. Flip to
    # false to trade safety for availability during a prolonged provider outage.
    arena_moderation_fail_closed: bool = True

    @model_validator(mode="after")
    def _placeholder_secrets_are_unset(self) -> Settings:
        """Warn on unrotated Secret Manager stubs and null the nullable ones."""
        logger = structlog.get_logger("coval_bench.config")
        for name, field in type(self).model_fields.items():
            value = getattr(self, name)
            raw = value.get_secret_value() if isinstance(value, SecretStr) else value
            if raw != SECRET_PLACEHOLDER:
                continue
            logger.warning("placeholder_secret", setting=name, env_var=name.upper())
            if type(None) in get_args(field.annotation):
                setattr(self, name, None)
        return self


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-cached Settings instance."""
    return Settings()
