# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS build

WORKDIR /app

# System deps required by audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy dependency manifests first for layer caching
# README.md is referenced by pyproject.toml's [project].readme and is required
# at metadata-validation time when uv builds the editable wheel for this package.
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies (no dev, no extras) into /app/.venv
RUN uv sync --frozen --no-dev

# Copy source tree + alembic config (db/cli.py reads /app/alembic.ini)
COPY src/ ./src/
COPY alembic.ini ./

# ---------------------------------------------------------------------------
# Final stage — slim, non-root (uid 65532)
#
# We use python:3.12-slim-bookworm rather than distroless because the
# `gcr.io/distroless/python3-debian12` image ships Python 3.11, which would
# break the 3.12 wheels installed in the build stage's venv. Slim still keeps
# the image well under the 500 MB budget while providing a matching 3.12
# runtime. We add a 65532 non-root user explicitly to preserve the
# unprivileged-by-default contract.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS final

LABEL org.opencontainers.image.source="https://github.com/coval-ai/benchmarks" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="coval-bench-runner"

# Runtime native libs for soundfile / librosa / ffmpeg-based codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 65532 nonroot \
    && useradd --uid 65532 --gid 65532 --no-create-home --shell /sbin/nologin nonroot

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"

COPY --from=build --chown=65532:65532 /app /app

USER 65532:65532

CMD ["python", "-m", "coval_bench", "run"]
