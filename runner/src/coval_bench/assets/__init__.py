# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from pydantic import BaseModel


class SecretRef(BaseModel, frozen=True, extra="forbid"):
    name: str
    purpose: str

    def resolve(self) -> str:
        value = os.environ.get(self.name)
        if not value:
            raise RuntimeError(f"{self.name} is unset. It holds {self.purpose}.")
        return value
