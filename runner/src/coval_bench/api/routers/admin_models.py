# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Admin registry endpoints: which models exist, their state, tags, and history.

Every route requires a coval-org Clerk session (``require_coval_admin``), and
every response is marked private: rows carry unpublished models and staff
emails, so a shared cache must never hold one. The tag vocabulary is public
and lives in ``tags.py``.

Writes enforce what the pydantic registry used to: the provider implementation
must exist for the modality (S2S is exempt; its data is fetched, not
synthesized here), tags must be in the vocabulary, a collected TTS model needs
a voice, and a non-empty voice pool is exactly one female and one male voice.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from psycopg_pool import AsyncConnectionPool

from coval_bench.api.clerk import CovalAdmin
from coval_bench.api.deps import get_pool, require_coval_admin
from coval_bench.api.internal import _RETIRED_BOARD_KEYS, never_shared
from coval_bench.api.schemas import (
    AdminChangeOut,
    AdminModelCreate,
    AdminModelOut,
    AdminModelPatch,
    AdminModelsResponse,
    AdminModelUpdateResponse,
)
from coval_bench.db.registry_store import (
    DuplicateKey,
    ModelChange,
    ModelRecord,
    NewModel,
    RegistryStore,
    StaleEdit,
)
from coval_bench.registries import Benchmark, Gender
from coval_bench.registries.metrics import METRIC_EXCLUSIONS
from coval_bench.registries.provider_keys import provider_names

router = APIRouter(tags=["admin"], dependencies=[Depends(require_coval_admin)])

# PATCH fields where null is a value, not an omission.
_NULLABLE_FIELDS = frozenset({"voice", "creator", "region"})


def _model_out(record: ModelRecord, history: list[ModelChange]) -> AdminModelOut:
    return AdminModelOut.model_validate(
        {
            **record.model_dump(),
            "history": [AdminChangeOut.model_validate(change.model_dump()) for change in history],
        }
    )


def _implemented_providers(modality: Benchmark) -> frozenset[str] | None:
    """Provider names with code for *modality*, or ``None`` when any name is fine."""
    if modality is Benchmark.STT:
        return provider_names("stt")
    if modality is Benchmark.TTS:
        return provider_names("tts")
    return None


async def _validate_model(candidate: NewModel | ModelRecord, store: RegistryStore) -> None:
    """The write-time rules the registry's review used to enforce; 422 on violation."""
    implemented = _implemented_providers(candidate.modality)
    if implemented is not None and candidate.provider not in implemented:
        raise HTTPException(
            422, f"no {candidate.modality} provider implementation named {candidate.provider!r}"
        )
    vocabulary = {tag.value for tag in await store.list_tags()}
    unknown = [tag for tag in candidate.tags if tag not in vocabulary]
    if unknown:
        raise HTTPException(422, f"unknown tags: {unknown}")
    if candidate.modality is not Benchmark.TTS and candidate.voice is not None:
        raise HTTPException(422, "a synthesis voice is TTS-only")
    if (
        candidate.modality is Benchmark.TTS
        and candidate.collected
        and candidate.voice is None
        and not candidate.voices
    ):
        raise HTTPException(422, "a collected TTS model needs a voice or a voice pool")
    if candidate.voices:
        if candidate.modality is not Benchmark.TTS:
            raise HTTPException(422, "voice pools are TTS-only")
        genders = sorted(voice.gender for voice in candidate.voices)
        if genders != [Gender.FEMALE, Gender.MALE]:
            raise HTTPException(422, "a voice pool holds exactly one female and one male voice")
        if len({voice.id for voice in candidate.voices}) != len(candidate.voices):
            raise HTTPException(422, "voice pool ids must be distinct")


def _rename_warnings(before: ModelRecord, after: ModelRecord) -> list[str]:
    """Code references a rename leaves pointing at the old or new key."""
    old = (before.provider, before.model)
    new = (after.provider, after.model)
    if old == new:
        return []
    warnings = [
        f"METRIC_EXCLUSIONS[{metric}] still names {old[0]}/{old[1]}; "
        "the renamed model is no longer excluded"
        for metric, pairs in METRIC_EXCLUSIONS.items()
        if old in pairs
    ]
    for retired, current in _RETIRED_BOARD_KEYS.items():
        if current == old:
            warnings.append(
                f"the retired board key {retired[0]}/{retired[1]} maps to {old[0]}/{old[1]}, "
                "which no longer exists"
            )
        if retired == new:
            warnings.append(
                f"{new[0]}/{new[1]} is a retired board key; artefacts stored under it "
                "are treated as embargoed history"
            )
    return warnings


@router.get("/admin/models", response_model=AdminModelsResponse)
async def list_admin_models(
    response: Response,
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> AdminModelsResponse:
    """Every registered model with its state, tags, and recent history."""
    never_shared(response)
    store = RegistryStore(pool)
    models = await store.list_models()
    history = await store.recent_history()
    return AdminModelsResponse(
        models=[_model_out(record, history.get(record.id, [])) for record in models]
    )


@router.post("/admin/models", response_model=AdminModelOut, status_code=201)
async def create_admin_model(
    body: AdminModelCreate,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> AdminModelOut:
    """Create a model. It starts collecting immediately; nothing is public yet."""
    never_shared(response)
    store = RegistryStore(pool)
    new = NewModel.model_validate(body.model_dump())
    await _validate_model(new, store)
    try:
        record = await store.insert_model(
            new, user_id=admin.user_id, org_id=None, email=admin.email
        )
    except DuplicateKey as exc:
        raise HTTPException(409, str(exc)) from exc
    return _model_out(record, await store.history(record.id))


@router.patch("/admin/models/{model_id}", response_model=AdminModelUpdateResponse)
async def update_admin_model(
    model_id: int,
    body: AdminModelPatch,
    response: Response,
    if_unmodified_since: str | None = Header(default=None),
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> AdminModelUpdateResponse:
    """Partially update a model; the history's ``old`` is what the editor saw."""
    never_shared(response)
    if if_unmodified_since is None:
        raise HTTPException(
            428, "If-Unmodified-Since is required: send the model's last-seen updated_at"
        )
    try:
        expected = datetime.fromisoformat(if_unmodified_since)
    except ValueError as exc:
        raise HTTPException(
            400, "If-Unmodified-Since must be the updated_at timestamp in ISO 8601"
        ) from exc
    if expected.tzinfo is None:
        raise HTTPException(400, "If-Unmodified-Since must carry a timezone")

    changes = body.model_dump(exclude_unset=True)
    null_forbidden = [
        field for field, value in changes.items() if value is None and field not in _NULLABLE_FIELDS
    ]
    if null_forbidden:
        raise HTTPException(422, f"fields that cannot be null: {null_forbidden}")

    store = RegistryStore(pool)
    current = await store.get_model(model_id)
    if current is None:
        raise HTTPException(404, "no such model")
    candidate = ModelRecord.model_validate({**current.model_dump(), **changes})
    await _validate_model(candidate, store)
    try:
        result = await store.update_model(
            model_id,
            changes,
            expected_updated_at=expected,
            user_id=admin.user_id,
            org_id=None,
            email=admin.email,
        )
    except StaleEdit as exc:
        raise HTTPException(
            412,
            f"the model changed at {exc.current.updated_at.isoformat()}; refetch and retry",
        ) from exc
    except DuplicateKey as exc:
        raise HTTPException(409, str(exc)) from exc
    if result is None:  # pragma: no cover — the row was read above
        raise HTTPException(404, "no such model")
    before, after = result
    history = await store.history(model_id)
    return AdminModelUpdateResponse(
        model=_model_out(after, history), warnings=_rename_warnings(before, after)
    )
