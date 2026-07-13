# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``GET /robots.txt`` — disallow-all robots exclusion file.

The API is served from a raw Cloud Run host that must stay out of search
indexes; the public site's robots.txt lives in the web frontend.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["robots"])

_ROBOTS_TXT = "User-agent: *\nDisallow: /\n"


@router.get("/robots.txt", include_in_schema=False)
async def robots_txt() -> PlainTextResponse:
    """Disallow all crawlers."""
    return PlainTextResponse(_ROBOTS_TXT)
