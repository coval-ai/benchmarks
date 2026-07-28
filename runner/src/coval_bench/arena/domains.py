# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The closed set of domains a battle can be tagged with.

Lives here rather than in ``api.schemas`` because it is arena vocabulary, not an API
shape: ``arena.prompts`` needs it, and importing it from ``api`` pulled in the whole
FastAPI app, which cycles back through ``api.routers.arena``.
"""

from __future__ import annotations

from typing import Literal

ArenaDomain = Literal["customer-service", "healthcare", "sales", "receptionist-booking", "other"]
"""Each value doubles as a leaderboard key, so the set is closed and excludes ``all`` —
that key is reserved for the aggregate board."""
