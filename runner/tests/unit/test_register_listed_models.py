# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for creating registry rows from the published model list."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from click.testing import CliRunner
from fastapi import HTTPException
from google.auth.exceptions import GoogleAuthError
from structlog.testing import capture_logs

from coval_bench import register_listed_models as module
from coval_bench.api.routers import admin_models
from coval_bench.api.routers.admin_models import create_model
from coval_bench.config import Settings
from coval_bench.db.registry_store import (
    DuplicateKey,
    ModelRecord,
    NewModel,
    RegistryStore,
    TagRecord,
)
from coval_bench.listed_models import ListedModel, parse_listed_models, read_listed_models
from coval_bench.registries import Benchmark
from coval_bench.registries.models import Architecture, Orchestration

_NOW = datetime(2026, 11, 25, tzinfo=UTC)


def _entry(name: str = "model-a", **registry: Any) -> dict[str, Any]:
    return {
        "provider": "acme",
        "name": name,
        "registry": {"architecture": "speech-to-speech", **registry},
    }


def _record(model_id: int, new: NewModel) -> ModelRecord:
    return ModelRecord.model_validate({**new.model_dump(), "id": model_id, "updated_at": _NOW})


class _Store(RegistryStore):
    def __init__(self, records: list[ModelRecord] | None = None, *, taken: bool = False) -> None:
        self.records = list(records or [])
        self.inserted: list[tuple[NewModel, str | None]] = []
        self._taken = taken

    async def list_models(self) -> list[ModelRecord]:
        return list(self.records)

    async def list_tags(self) -> list[TagRecord]:
        return []

    async def insert_model(
        self, new: NewModel, *, user_id: str | None, org_id: str | None, email: str | None
    ) -> ModelRecord:
        if self._taken:
            raise DuplicateKey("taken")
        record = _record(len(self.records) + 1, new)
        self.records.append(record)
        self.inserted.append((new, user_id))
        return record


def _listed(*entries: dict[str, Any]) -> list[ListedModel]:
    models, problems = parse_listed_models({"s2s": list(entries)})
    assert problems == []
    return models


def _register(store: _Store, *entries: dict[str, Any]) -> module.Registered:
    return asyncio.run(module.register_models(_listed(*entries), store, Settings()))


def test_a_valid_entry_parses_with_the_harness_default() -> None:
    models, problems = parse_listed_models({"s2s": [_entry()]})
    assert problems == []
    assert models[0].registry.orchestration is Orchestration.HARNESS


def test_a_bad_entry_is_a_problem_and_the_rest_still_parse() -> None:
    no_architecture = {"provider": "acme", "name": "secret-b", "registry": {}}
    unknown_field = _entry("secret-c", display_name="Secret C")
    models, problems = parse_listed_models({"s2s": [_entry(), no_architecture, unknown_field]})
    assert [model.name for model in models] == ["model-a"]
    assert problems == [
        "s2s[1].registry.architecture: missing",
        "s2s[2].registry: unknown field",
    ]
    assert not any("secret" in problem or "acme" in problem for problem in problems)


def test_a_problem_never_repeats_a_key_or_an_entry_that_is_not_a_table() -> None:
    misnested = {**_entry(), "secret-model": {}}
    _, problems = parse_listed_models({"s2s": [misnested, "secret-text"]})
    assert problems == ["s2s[0]: unknown field", "s2s[1]: model_type"]


@pytest.mark.parametrize("raw", [{}, {"s2s": {}}, []])
def test_a_list_without_s2s_entries_is_refused(raw: object) -> None:
    with pytest.raises(ValueError, match="no s2s list"):
        parse_listed_models(raw)


def test_an_address_that_is_not_a_bucket_path_is_refused_without_repeating_it() -> None:
    with pytest.raises(ValueError) as caught:
        read_listed_models("gs://bucket-only")
    assert "bucket-only" not in str(caught.value)


def test_a_missing_row_is_created_for_coval_only() -> None:
    store = _Store()
    done = _register(store, _entry())
    assert done == module.Registered(created=(1,), existing=0, refused=())
    new, user_id = store.inserted[0]
    assert (new.modality, new.provider, new.model) == (Benchmark.S2S, "acme", "model-a")
    assert (new.visibility, new.collected) == ("coval", True)
    assert (new.architecture, new.orchestration) == (
        Architecture.SPEECH_TO_SPEECH,
        Orchestration.HARNESS,
    )
    assert (new.display_name, new.creator) == (None, None)
    assert user_id == "register-listed-models"


def test_an_existing_row_is_left_as_it_is() -> None:
    existing = _record(
        7,
        NewModel(
            modality=Benchmark.S2S,
            provider="acme",
            model="model-a",
            visibility="public",
            collected=False,
            architecture=Architecture.SPEECH_TO_SPEECH,
        ),
    )
    store = _Store([existing])
    done = _register(store, _entry(orchestration="platform"))
    assert done == module.Registered(created=(), existing=1, refused=())
    assert store.inserted == []
    assert store.records == [existing]


def test_a_same_named_row_of_another_kind_does_not_count_as_existing() -> None:
    tts = _record(3, NewModel(modality=Benchmark.TTS, provider="acme", model="model-a", voice="v"))
    done = _register(_Store([tts]), _entry())
    assert done.created == (2,)


def test_a_model_listed_twice_gets_one_row() -> None:
    store = _Store()
    done = _register(store, _entry(), _entry())
    assert done == module.Registered(created=(1,), existing=1, refused=())
    assert len(store.inserted) == 1


def test_a_row_created_by_someone_else_first_counts_as_existing() -> None:
    done = _register(_Store(taken=True), _entry())
    assert done == module.Registered(created=(), existing=1, refused=())


def test_a_refusal_keeps_its_status_and_the_rest_are_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    async def create(new: NewModel, *args: Any, **kwargs: Any) -> ModelRecord:
        if new.model == "model-a":
            raise HTTPException(422, "model-a is not allowed")
        return await create_model(new, *args, **kwargs)

    monkeypatch.setattr(admin_models, "create_model", create)
    done = _register(_Store(), _entry(), _entry("model-b"))
    assert done == module.Registered(created=(1,), existing=0, refused=("s2s[0]: 422",))


def test_creating_a_row_never_starts_the_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []

    def start(job: str) -> bool:
        started.append(job)
        return True

    monkeypatch.setattr(admin_models, "start_s2s_sync", start)
    done = asyncio.run(
        module.register_models(_listed(_entry()), _Store(), Settings(s2s_sync_job="s2s-sync"))
    )
    assert done.created == (1,)
    assert started == []


def test_a_database_failure_partway_reports_the_rows_already_created() -> None:
    class _Failing(_Store):
        async def insert_model(
            self, new: NewModel, *, user_id: str | None, org_id: str | None, email: str | None
        ) -> ModelRecord:
            if new.model == "model-b":
                raise psycopg.OperationalError("connection lost")
            return await super().insert_model(new, user_id=user_id, org_id=org_id, email=email)

    with pytest.raises(module.Stopped) as caught:
        _register(_Failing(), _entry(), _entry("model-b"))
    assert caught.value.created == (1,)


def _invoke(
    monkeypatch: pytest.MonkeyPatch, store: _Store, raw: object
) -> tuple[int, str, list[Any]]:
    @asynccontextmanager
    async def pool(settings: Settings) -> AsyncIterator[None]:
        yield None

    def read(uri: str) -> tuple[list[ListedModel], list[str]]:
        return parse_listed_models(raw)

    monkeypatch.setattr(module, "get_settings", lambda: Settings())
    monkeypatch.setattr(module, "configure_logging", lambda **_: None)
    monkeypatch.setattr(module, "lifespan_pool", pool)
    monkeypatch.setattr(module, "RegistryStore", lambda _pool: store)
    monkeypatch.setattr(module, "read_listed_models", read)
    with capture_logs() as logs:
        result = CliRunner().invoke(module.register_listed_models, [])
    return result.exit_code, result.output, logs


def test_an_empty_list_exits_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    code, _, logs = _invoke(monkeypatch, _Store(), {"s2s": []})
    assert code == 0
    assert logs[-1]["event"] == "listed_models_registered"
    assert (logs[-1]["created"], logs[-1]["existing"]) == ([], 0)


def test_a_bad_entry_stops_the_command_before_any_row_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    raw = {"s2s": [_entry("secret-a"), {"provider": "acme", "name": "secret-b", "registry": {}}]}
    code, output, logs = _invoke(monkeypatch, store, raw)
    assert code == 1
    assert store.inserted == []
    assert logs == [
        {
            "event": "RUN_FAILED",
            "invalid": ["s2s[1].registry.architecture: missing"],
            "log_level": "error",
        }
    ]
    assert "secret" not in output


def test_a_refusal_fails_the_command_and_says_which_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def create(new: NewModel, *args: Any, **kwargs: Any) -> ModelRecord:
        raise HTTPException(422, "secret-a is not allowed")

    monkeypatch.setattr(admin_models, "create_model", create)
    code, output, logs = _invoke(monkeypatch, _Store(), {"s2s": [_entry("secret-a")]})
    assert code == 1
    assert logs[-1] == {"event": "RUN_FAILED", "refused": ["s2s[0]: 422"], "log_level": "error"}
    assert "secret" not in output + repr(logs)


@pytest.mark.parametrize(
    "error",
    [
        GoogleAuthError("token for sa@private-bucket"),  # type: ignore[no-untyped-call]
        RuntimeError("gs://private-bucket"),
    ],
)
def test_a_failed_read_names_no_address(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def read(uri: str) -> tuple[list[ListedModel], list[str]]:
        raise error

    monkeypatch.setattr(module, "get_settings", lambda: Settings())
    monkeypatch.setattr(module, "configure_logging", lambda **_: None)
    monkeypatch.setattr(module, "read_listed_models", read)
    with capture_logs() as logs:
        result = CliRunner().invoke(module.register_listed_models, [])
    assert result.exit_code == 1
    assert logs == [{"event": "RUN_FAILED", "error": type(error).__name__, "log_level": "error"}]
    assert "private-bucket" not in result.output
