# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""FastAPI service for the Coval Benchmarks public API.

This package is intentionally co-located with the runner so that the DB writer
schema and reader queries can never drift (ADR-014).

Usage::

    from coval_bench.api import create_app
    app = create_app()
"""

from coval_bench.api.app import create_app

__all__ = ["create_app"]
