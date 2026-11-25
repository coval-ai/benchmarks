# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The model list infra publishes for the registry: which models should have a row."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from coval_bench import gcs
from coval_bench.registries.models import Architecture, Orchestration

_Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _RegistryDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: Architecture
    orchestration: Orchestration = Orchestration.HARNESS


class ListedModel(BaseModel):
    """One S2S model on the list, with what its registry row needs."""

    model_config = ConfigDict(extra="forbid")

    provider: _Name
    name: _Name
    registry: _RegistryDetails


def parse_listed_models(raw: object) -> tuple[list[ListedModel], list[str]]:
    """The list's valid S2S entries, and one problem per entry that isn't.

    A problem names the entry's position and field, never a value from the list.
    """
    entries = raw.get("s2s") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise ValueError("the model list has no s2s list")
    models: list[ListedModel] = []
    problems: list[str] = []
    for index, entry in enumerate(entries):
        try:
            models.append(ListedModel.model_validate(entry))
        except ValidationError as exc:
            problems.extend(
                f"s2s[{index}].{'.'.join(str(part) for part in error['loc'])}: {error['type']}"
                for error in exc.errors()
            )
    return models, problems


def read_listed_models(uri: str) -> tuple[list[ListedModel], list[str]]:
    """Parse the list at ``uri`` (``gs://bucket/key``)."""
    bucket, _, key = uri.removeprefix("gs://").partition("/")
    if not uri.startswith("gs://") or not bucket or not key:
        raise ValueError("the model list address is not a gs://bucket/key URI")
    raw = gcs.read_json(bucket, key)
    if raw is None:
        raise RuntimeError("the model list was not found")
    return parse_listed_models(raw)
