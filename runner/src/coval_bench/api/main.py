# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Uvicorn entrypoint for the Coval Benchmarks API.

Production startup::

    uvicorn coval_bench.api.main:app --host 0.0.0.0 --port 8080

Or via the CLI (wired by the orchestrator agent in ``__main__.py``)::

    coval-bench serve

The ``app`` module-level object is required so that uvicorn can import it as
``coval_bench.api.main:app``.
"""

from __future__ import annotations

import os

from coval_bench.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "coval_bench.api.main:app",
        host="0.0.0.0",  # noqa: S104
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
