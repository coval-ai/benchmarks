# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Serve timeline percentiles from closed 1h/4h rollups; drop dirty-hour tracking."""

from __future__ import annotations

from alembic import op

revision = "20261008_0044"
down_revision = "20261005_0043"
branch_labels = None
depends_on = None

_FINITE = "NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)"


def upgrade() -> None:
    op.execute(f"""
    ALTER TABLE benchmarks_v2.metric_values_by_bucket
      ADD COLUMN p90 DOUBLE PRECISION CHECK (p90 {_FINITE}),
      ADD COLUMN p95 DOUBLE PRECISION CHECK (p95 {_FINITE}),
      ADD CHECK ((p90 IS NULL) = (p95 IS NULL)),
      ADD CHECK (p90 IS NULL OR (p75 <= p90 AND p90 <= p95 AND p95 <= max_value));

    CREATE TABLE benchmarks_v2.dashboard_bucket_aggregates (
      provider TEXT NOT NULL CHECK (provider <> ''),
      model TEXT NOT NULL CHECK (model <> ''),
      benchmark TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S', 'LLM')),
      dataset_id TEXT NOT NULL CHECK (dataset_id <> ''),
      metric_id BIGINT NOT NULL REFERENCES benchmarks_v2.metrics(id) ON DELETE RESTRICT,
      metric_version TEXT NOT NULL CHECK (metric_version <> ''),
      evaluation_variant TEXT NOT NULL CHECK (evaluation_variant <> ''),
      value_key TEXT NOT NULL CHECK (value_key <> ''),
      unit TEXT NOT NULL CHECK (unit <> ''),
      interval_seconds INTEGER NOT NULL CHECK (interval_seconds IN (3600, 14400)),
      bucket_at TIMESTAMPTZ NOT NULL,
      min_value DOUBLE PRECISION NOT NULL CHECK (min_value {_FINITE}),
      p25 DOUBLE PRECISION NOT NULL CHECK (p25 {_FINITE}),
      p50 DOUBLE PRECISION NOT NULL CHECK (p50 {_FINITE}),
      p75 DOUBLE PRECISION NOT NULL CHECK (p75 {_FINITE}),
      p90 DOUBLE PRECISION NOT NULL CHECK (p90 {_FINITE}),
      p95 DOUBLE PRECISION NOT NULL CHECK (p95 {_FINITE}),
      max_value DOUBLE PRECISION NOT NULL CHECK (max_value {_FINITE}),
      value_sum DOUBLE PRECISION NOT NULL CHECK (value_sum {_FINITE}),
      sample_count INTEGER NOT NULL CHECK (sample_count > 0),
      latest_source_at TIMESTAMPTZ NOT NULL,
      CHECK (min_value <= p25 AND p25 <= p50 AND p50 <= p75 AND p75 <= p90
             AND p90 <= p95 AND p95 <= max_value),
      CHECK (mod(extract(epoch FROM bucket_at)::bigint, interval_seconds) = 0),
      PRIMARY KEY (provider, model, benchmark, dataset_id, metric_id, metric_version,
                   evaluation_variant, value_key, interval_seconds, bucket_at)
    );
    CREATE INDEX dashboard_bucket_aggregates_lookup
      ON benchmarks_v2.dashboard_bucket_aggregates (benchmark, dataset_id, interval_seconds, bucket_at);

    CREATE TABLE benchmarks_v2.dashboard_bucket_fills (
      interval_seconds INTEGER NOT NULL CHECK (interval_seconds IN (3600, 14400)),
      bucket_at TIMESTAMPTZ NOT NULL,
      filled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (interval_seconds, bucket_at)
    );

    DROP TABLE benchmarks_v2.dashboard_hourly_state;
    DROP TABLE benchmarks_v2.dashboard_source_refreshes;
    DROP TABLE benchmarks_v2.dashboard_hourly_aggregates;

    DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api') THEN
      GRANT SELECT ON benchmarks_v2.dashboard_bucket_aggregates,
                      benchmarks_v2.dashboard_bucket_fills TO api;
    END IF; END $$;
    """)


def downgrade() -> None:
    op.execute("""
    DROP TABLE benchmarks_v2.dashboard_bucket_fills;
    DROP TABLE benchmarks_v2.dashboard_bucket_aggregates;
    ALTER TABLE benchmarks_v2.metric_values_by_bucket DROP COLUMN p90, DROP COLUMN p95;

    CREATE TABLE benchmarks_v2.dashboard_hourly_aggregates (
      provider TEXT NOT NULL, model TEXT NOT NULL, benchmark TEXT NOT NULL, dataset_id TEXT NOT NULL,
      metric_id BIGINT NOT NULL,
      metric_type TEXT NOT NULL, metric_version TEXT NOT NULL, evaluation_variant TEXT NOT NULL,
      hour_at TIMESTAMPTZ NOT NULL CHECK (hour_at = date_trunc('hour', hour_at, 'UTC')),
      primary_sum DOUBLE PRECISION NOT NULL, sample_count BIGINT NOT NULL CHECK (sample_count > 0),
      numerator_sum DOUBLE PRECISION, denominator_sum DOUBLE PRECISION, coverage_complete BOOLEAN NOT NULL,
      source_count BIGINT NOT NULL CHECK (source_count > 0), latest_source_at TIMESTAMPTZ,
      definition_revision INTEGER NOT NULL,
      metadata JSONB NOT NULL DEFAULT '{"schema_version" : 1}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
      PRIMARY KEY (provider, model, benchmark, dataset_id, metric_type, metric_version, evaluation_variant, hour_at)
    );
    -- Names match migrations 0035/0043 so their downgrades keep working.
    ALTER TABLE benchmarks_v2.dashboard_hourly_aggregates
      ADD CONSTRAINT dashboard_hourly_aggregates_metric_id_fkey
      FOREIGN KEY (metric_id) REFERENCES benchmarks_v2.metrics(id) ON DELETE RESTRICT;
    CREATE TRIGGER dashboard_hourly_sync_metric_identity
      BEFORE INSERT OR UPDATE OF metric_type, metric_id
      ON benchmarks_v2.dashboard_hourly_aggregates FOR EACH ROW
      EXECUTE FUNCTION benchmarks_v2.sync_dashboard_metric_identity();
    CREATE UNIQUE INDEX dashboard_hourly_aggregates_metric_identity_key
      ON benchmarks_v2.dashboard_hourly_aggregates
      (provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant, hour_at);
    CREATE INDEX dashboard_hourly_aggregates_hour_idx ON benchmarks_v2.dashboard_hourly_aggregates (hour_at);
    CREATE INDEX dashboard_hourly_aggregates_lookup ON benchmarks_v2.dashboard_hourly_aggregates (benchmark, dataset_id, hour_at);
    CREATE TABLE benchmarks_v2.dashboard_hourly_state (
      hour_at TIMESTAMPTZ PRIMARY KEY CHECK (hour_at = date_trunc('hour', hour_at, 'UTC')),
      dirty BOOLEAN NOT NULL, refreshed_at TIMESTAMPTZ,
      definition_revision INTEGER NOT NULL, definition_fingerprint TEXT NOT NULL,
      metadata JSONB NOT NULL DEFAULT '{"schema_version" : 1}'::jsonb CHECK (jsonb_typeof(metadata) = 'object')
    );
    CREATE TABLE benchmarks_v2.dashboard_source_refreshes (
      bucket_at TIMESTAMPTZ PRIMARY KEY,
      requested_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api') THEN
      GRANT SELECT ON benchmarks_v2.dashboard_hourly_aggregates, benchmarks_v2.dashboard_hourly_state,
                      benchmarks_v2.dashboard_source_refreshes TO api;
    END IF; END $$;
    """)
