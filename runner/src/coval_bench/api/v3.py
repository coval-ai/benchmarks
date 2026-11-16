# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The error shape every ``/v3`` endpoint answers with."""

from __future__ import annotations

from typing import Any

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

PREFIX = "/v3"


_STATUS_CODES = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    429: "rate_limited",
    503: "unavailable",
}


def error_body(status: int, message: str, param: str | None = None) -> dict[str, Any]:
    """The body for an HTTP error, with the code its status maps to."""
    code = _STATUS_CODES.get(status, "error")
    return {"error": {"code": code, "message": message, "param": param}}


def http_error_body(exc: StarletteHTTPException) -> dict[str, Any]:
    """The body for an HTTP error a route or the router raised."""
    return error_body(exc.status_code, str(exc.detail))


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
