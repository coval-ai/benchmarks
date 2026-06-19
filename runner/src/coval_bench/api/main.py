# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Uvicorn entrypoint for the Coval Benchmarks API.

Production startup (see ``Dockerfile.api``)::

    uvicorn coval_bench.api.main:app --host 0.0.0.0 --port 8080

Importing this module configures logging as a side effect, so JSON output is in
effect no matter how uvicorn is launched: uvicorn always imports the app *after*
installing its own (default) logging, so our ``dictConfig`` below wins. This
avoids tying the logging setup to a particular CMD form.

The ``app`` module-level object is required so that uvicorn can import it as
``coval_bench.api.main:app``.
"""

from __future__ import annotations

import logging.config
import os

from coval_bench.api.app import create_app
from coval_bench.config import get_settings
from coval_bench.logging import configure_logging, uvicorn_log_config

_log_level = get_settings().log_level
configure_logging(level=_log_level)
logging.config.dictConfig(uvicorn_log_config(_log_level))

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "coval_bench.api.main:app",
        host="0.0.0.0",  # noqa: S104
        port=int(os.environ.get("PORT", "8080")),
        log_level=_log_level.lower(),
        log_config=uvicorn_log_config(_log_level),
        access_log=False,
    )
