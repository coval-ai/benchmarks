# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Create the registry rows the published model list is missing."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import click
import psycopg
import structlog
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError

from coval_bench.config import Settings, get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.registry_store import NewModel, RegistryStore
from coval_bench.listed_models import ListedModel, read_listed_models
from coval_bench.logging import configure_logging
from coval_bench.registries import Benchmark

logger = structlog.get_logger("coval_bench.register_listed_models")

_ACTOR = "register-listed-models"


@dataclass(frozen=True)
class Registered:
    """What one pass did: the new rows' ids, how many were there, and each refusal."""

    created: tuple[int, ...]
    existing: int
    refused: tuple[str, ...]


class Stopped(Exception):
    """The database failed partway; ``created`` holds the rows made before it did."""

    def __init__(self, created: tuple[int, ...]) -> None:
        super().__init__("the database failed partway")
        self.created = created


async def register_models(
    listed: Sequence[ListedModel], store: RegistryStore, settings: Settings
) -> Registered:
    """Create a row for each listed model that has none; an existing row is left as it is."""
    from fastapi import HTTPException

    from coval_bench.api.routers.admin_models import create_model

    quiet = settings.model_copy(update={"s2s_sync_job": ""})
    have = {
        (record.provider, record.model)
        for record in await store.list_models()
        if record.modality is Benchmark.S2S
    }
    created: list[int] = []
    refused: list[str] = []
    existing = 0
    for index, entry in enumerate(listed):
        key = (entry.provider, entry.name)
        if key in have:
            existing += 1
            continue
        new = NewModel(
            modality=Benchmark.S2S,
            provider=entry.provider,
            model=entry.name,
            visibility="coval",
            architecture=entry.registry.architecture,
            orchestration=entry.registry.orchestration,
        )
        try:
            record = await create_model(new, store, quiet, org_id=None, user_id=_ACTOR)
        except HTTPException as exc:
            if exc.status_code == 409:
                existing += 1
            else:
                refused.append(f"s2s[{index}]: {exc.status_code}")
            continue
        except psycopg.Error as exc:
            raise Stopped(tuple(created)) from exc
        have.add(key)
        created.append(record.id)
    return Registered(tuple(created), existing, tuple(refused))


@click.command(name="register-listed-models")
def register_listed_models() -> None:
    """Create a registry row for each model on the published list that has none.

    New rows are S2S, collected and visible to Coval only. Existing rows are never changed.
    An entry the list gets wrong stops the run before any row is created.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level)

    async def _register(listed: Sequence[ListedModel]) -> Registered:
        async with lifespan_pool(settings) as pool:
            return await register_models(listed, RegistryStore(pool), settings)

    try:
        listed, problems = read_listed_models(settings.listed_models_uri.strip())
        if problems:
            logger.error("RUN_FAILED", invalid=problems)
            raise click.ClickException(f"{len(problems)} invalid; nothing was created")
        done = asyncio.run(_register(listed))
    except Stopped as exc:
        logger.error("RUN_FAILED", error=type(exc.__cause__).__name__, created=list(exc.created))
        raise click.ClickException("the listed models were not all registered") from exc
    except (ValueError, RuntimeError, psycopg.Error, GoogleAPIError, GoogleAuthError) as exc:
        logger.error("RUN_FAILED", error=type(exc).__name__)
        raise click.ClickException("the listed models were not registered") from exc
    logger.info("listed_models_registered", created=list(done.created), existing=done.existing)
    if done.refused:
        logger.error("RUN_FAILED", refused=list(done.refused))
        raise click.ClickException(f"{len(done.refused)} refused; see the log")
