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

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    database_url: str

    # --- Dataset ---
    dataset_bucket: str = "coval-benchmarks-datasets"
    dataset_id: str = "stt-v1"

    # --- Runner ---
    runner_sha: str = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Provider API keys (all optional; loaded from Secret Manager at runtime) ---
    openai_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None
    cartesia_api_key: SecretStr | None = None
    deepgram_api_key: SecretStr | None = None
    assemblyai_api_key: SecretStr | None = None
    speechmatics_api_key: SecretStr | None = None
    hume_api_key: SecretStr | None = None
    rime_api_key: SecretStr | None = None

    # Path to a Google service-account JSON file mounted as a Secret-as-volume.
    google_application_credentials: Path | None = None

    # GCP project ID hosting the Google STT v2 recognizer. Required only when
    # the Google STT provider is enabled (optional `google-stt` extra).
    google_project_id: str | None = None

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


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-cached Settings instance."""
    return Settings()
