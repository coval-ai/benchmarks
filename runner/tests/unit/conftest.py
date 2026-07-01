# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for unit tests."""

from pytest_postgresql.factories import postgresql_proc

# One embedded Postgres server (random free port) for all DB-backed unit
# tests; each client fixture still gets a clean per-test database.
pg_proc = postgresql_proc(port=None)
