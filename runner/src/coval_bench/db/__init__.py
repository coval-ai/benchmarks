# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Database layer: async psycopg3 pool, Pydantic domain models, RunWriter."""

from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.conn import get_pool, lifespan_pool
from coval_bench.db.models import (
    Battle,
    Benchmark,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    Vote,
    VoteOutcome,
    VoterType,
)
from coval_bench.db.writer import RunWriter

__all__ = [
    "get_pool",
    "lifespan_pool",
    "ArenaStore",
    "Battle",
    "Benchmark",
    "Result",
    "ResultStatus",
    "Run",
    "RunStatus",
    "RunWriter",
    "Vote",
    "VoteOutcome",
    "VoterType",
]
