# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The error shape every ``/v3`` endpoint answers with."""

from __future__ import annotations

from typing import Any

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

PREFIX = "/v3"


def http_error_body(exc: StarletteHTTPException) -> dict[str, Any]:
    """The body for an HTTP error."""
    code = "unavailable" if exc.status_code == 503 else "error"
    return {"error": {"code": code, "message": str(exc.detail), "param": None}}


def validation_error_body(exc: RequestValidationError) -> dict[str, Any]:
    """The body for a parameter FastAPI rejected, naming the parameter."""
    first = exc.errors()[0]
    return {
        "error": {
            "code": "invalid_parameter",
            "message": first["msg"],
            "param": str(first["loc"][-1]),
        }
    }
