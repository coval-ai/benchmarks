# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The error shape every ``/v3`` endpoint answers with."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from coval_bench.registries import Benchmark

PREFIX = "/v3"
MODALITY_PREFIX = PREFIX + "/{modality}"

_MODALITIES = {benchmark.value.lower(): benchmark for benchmark in Benchmark}


def path_modality(modality: str) -> Benchmark:
    """The modality a results path names, lowercase: ``/v3/stt/quality``."""
    found = _MODALITIES.get(modality)
    if found is None:
        raise HTTPException(404, f"no such modality; use one of {sorted(_MODALITIES)}")
    return found


_STATUS_CODES = {
    400: "invalid_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    412: "precondition_failed",
    428: "precondition_required",
    429: "rate_limited",
    503: "unavailable",
}


def error_body(status: int, message: str, param: str | None = None) -> dict[str, Any]:
    """The body for an HTTP error, with the code its status maps to."""
    code = _STATUS_CODES.get(status, "error")
    return {"error": {"code": code, "message": message, "param": param}}


def http_status(exc: StarletteHTTPException) -> int:
    """The status an HTTP error is answered with: a request ``/v3`` won't take is a 400."""
    return 400 if exc.status_code == 422 else exc.status_code


def http_error_body(exc: StarletteHTTPException) -> dict[str, Any]:
    """The body for an HTTP error a route or the router raised."""
    return error_body(http_status(exc), str(exc.detail))


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
