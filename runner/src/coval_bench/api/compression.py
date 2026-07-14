# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gzip compression for API responses, bypassing static clip files.

Wraps Starlette's ``GZipMiddleware`` with a path-prefix bypass. Compressing
the ``/clips`` static mount would break it: a 206 body gets gzipped while
``Content-Range`` still describes offsets in the original file, and streamed
file responses lose ``Content-Length``. WAV clips barely compress anyway.
"""

from __future__ import annotations

from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


class SelectiveGZipMiddleware(GZipMiddleware):
    """``GZipMiddleware`` that leaves excluded path prefixes uncompressed."""

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int = 500,
        compresslevel: int = 9,
        exclude_prefixes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(app, minimum_size, compresslevel)
        self.exclude_prefixes = exclude_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(self.exclude_prefixes):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)
