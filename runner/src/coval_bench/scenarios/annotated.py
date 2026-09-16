# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The contract-model base: required fields, plus ``_``-prefixed rationale keys."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class AnnotatedModel(BaseModel):
    """Accept ``_``-prefixed extras as rationale; reject any other unknown key."""

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="after")
    def _only_underscore_extras(self) -> AnnotatedModel:
        extras = self.__pydantic_extra__ or {}
        unknown = sorted(k for k in extras if not k.startswith("_"))
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in {type(self).__name__}; "
                "rationale keys must start with '_'"
            )
        return self
