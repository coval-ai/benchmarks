# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for pricing-collector tests."""

from pytest_postgresql.factories import postgresql_proc

# Same convention as tests/unit: one embedded Postgres server for the package,
# a clean per-test database per client fixture.
pg_proc = postgresql_proc(port=None)
