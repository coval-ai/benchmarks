# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""STT sanity check without Postgres: packaged manifest + GCS fetch + one provider.

``DATABASE_URL`` is not used — a placeholder URL satisfies :class:`Settings` validation.
Provider API keys still come from the environment / dotenv files (same as production).

Reads dotenv files below, **merging** them so **later paths override** earlier
ones (same as typical ``.env`` + ``.env.local``). Merged keys are applied with
``os.environ.setdefault`` so **variables you already exported in the shell** still
win. Paths: repo ``.env``, ``runner/.env``, repo ``.env.local``,
``runner/.env.local``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any

import structlog
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.datasets.loader import load_stt_dataset
from coval_bench.metrics import compute_wer
from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt import STT_PROVIDERS
from coval_bench.runner.retry import with_retry

_log = structlog.get_logger(__name__)

# Sentinel only — nothing connects when running preview (no orchestrator / pool).
_PREVIEW_DB_SENTINEL = (
    "postgresql://preview-not-used:preview-not-used@127.0.0.1:65432/preview_not_used"
)

_STT_TIMEOUT_S = 45

# runner/src/coval_bench/stt_preview.py -> parents[2] == runner/
_RUNNER_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _RUNNER_ROOT.parent


def _parse_dotenv_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def _load_preview_env_files() -> None:
    """Populate ``os.environ`` from common dotenv paths (do not override existing)."""
    candidates = (
        _REPO_ROOT / ".env",
        _RUNNER_ROOT / ".env",
        _REPO_ROOT / ".env.local",
        _RUNNER_ROOT / ".env.local",
    )
    merged: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        merged.update(_parse_dotenv_text(text))
    for key, val in merged.items():
        os.environ.setdefault(key, val)


def preview_settings(*, dataset_id: str, dataset_bucket: str) -> Settings:
    """Settings row for loader + provider keys; DB URL is ignored by this flow."""
    _load_preview_env_files()
    return Settings(
        database_url=os.environ.get("DATABASE_URL", _PREVIEW_DB_SENTINEL),
        dataset_bucket=dataset_bucket,
        dataset_id=dataset_id,
    )


def instantiate_stt(provider: str, model: str, settings: Settings) -> STTProvider | None:
    """Return provider instance or ``None`` if *provider* is unknown."""
    provider_cls = STT_PROVIDERS.get(provider)
    if provider_cls is None:
        return None
    kwargs: dict[str, Any] = {"model": model}
    if provider == "google":
        # Google STT uses ADC (``GOOGLE_APPLICATION_CREDENTIALS``); ``api_key`` on the
        # provider is unused but still a required constructor arg.
        if settings.google_project_id is None:
            raise RuntimeError("google_project_id is required for google STT preview")
        kwargs["project_id"] = settings.google_project_id
        kwargs["api_key"] = SecretStr("unused")
    else:
        key_attr = f"{provider}_api_key"
        api_key = getattr(settings, key_attr, None)
        if api_key is None:
            msg = (
                f"Set {key_attr} in the environment or .env — preview needs a real API key "
                f"for {provider!r}."
            )
            raise RuntimeError(msg)
        kwargs["api_key"] = api_key
    return provider_cls(**kwargs)


async def run_stt_preview(
    *,
    dataset_id: str,
    dataset_bucket: str,
    provider: str,
    model: str,
    limit: int,
) -> tuple[int, dict[str, Any]]:
    """Load up to *limit* items from GCS, transcribe with *provider*.

    Returns ``(exit_code, payload_dict)`` — exit ``0`` all items OK, ``1`` partial/total
    failure, ``2`` unknown provider name.
    """
    settings = preview_settings(dataset_id=dataset_id, dataset_bucket=dataset_bucket)

    try:
        inst = instantiate_stt(provider, model, settings)
    except RuntimeError as exc:
        return 1, {"event": "stt_preview", "ok": False, "error": str(exc)}

    if inst is None:
        return (
            2,
            {
                "event": "stt_preview",
                "ok": False,
                "error": "unknown_provider",
                "provider": provider,
                "known": sorted(STT_PROVIDERS.keys()),
            },
        )

    _log.info(
        "stt_preview_loading_dataset",
        dataset_id=dataset_id,
        bucket=dataset_bucket,
        limit=limit,
    )
    dataset = load_stt_dataset(dataset_id, settings=settings)

    items_out: list[dict[str, Any]] = []
    all_ok = True

    for item in dataset.items[:limit]:
        row: dict[str, Any] = {"audio_filename": item.path.name, "path": str(item.path)}
        pcm_blob = item.path.read_bytes()
        dur_sec = item.duration_sec

        try:
            async with asyncio.timeout(_STT_TIMEOUT_S):

                async def _measure(
                    blob: bytes = pcm_blob,
                    dur: float = dur_sec,
                ) -> TranscriptionResult:
                    return await inst.measure_ttft(blob, 1, 2, 16000, 0.1, dur)

                tr = await with_retry(_measure)
        except Exception as exc:
            all_ok = False
            row["ok"] = False
            row["error"] = str(exc)
            items_out.append(row)
            continue

        if tr.error:
            all_ok = False
            row["ok"] = False
            row["error"] = tr.error
        else:
            row["ok"] = True
            row["ttft_seconds"] = tr.ttft_seconds
            row["audio_to_final_seconds"] = tr.audio_to_final_seconds
            row["transcript"] = tr.complete_transcript
            ref = item.transcript
            row["reference"] = ref
            wer_pct: float | None = None
            if ref and tr.complete_transcript:
                with contextlib.suppress(Exception):
                    wer_pct = compute_wer(ref, tr.complete_transcript).wer_percentage
            row["wer_percentage"] = wer_pct

        items_out.append(row)

    payload = {
        "event": "stt_preview",
        "dataset_id": dataset_id,
        "dataset_bucket": dataset_bucket,
        "provider": provider,
        "model": model,
        "items": items_out,
        "ok": all_ok,
    }
    return (0 if all_ok else 1), payload
