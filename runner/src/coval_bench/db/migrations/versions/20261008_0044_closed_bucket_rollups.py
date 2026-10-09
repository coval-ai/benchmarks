# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""One rollups table at run, 1h and 4h grains, fed by a queue of changed run slots."""

from __future__ import annotations

from alembic import op

revision = "20261008_0044"
down_revision = "20261005_0043"
branch_labels = None
depends_on = None

_FINITE = "NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)"


def upgrade() -> None:
    op.execute(f"""
    CREATE TABLE benchmarks_v2.dashboard_rollups (
      provider TEXT NOT NULL CHECK (provider <> ''),
      model TEXT NOT NULL CHECK (model <> ''),
      benchmark TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S', 'LLM')),
      dataset_id TEXT NOT NULL CHECK (dataset_id <> ''),
      metric_id BIGINT NOT NULL REFERENCES benchmarks_v2.metrics(id) ON DELETE RESTRICT,
      metric_version TEXT NOT NULL CHECK (metric_version <> ''),
      evaluation_variant TEXT NOT NULL CHECK (evaluation_variant <> ''),
      value_key TEXT NOT NULL CHECK (value_key IN ('primary', 'roundtrip', 'leading_silence')),
      grain TEXT NOT NULL CHECK (grain IN ('run', '1h', '4h')),
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
      wer_error_words DOUBLE PRECISION CHECK (wer_error_words {_FINITE}),
      wer_reference_words DOUBLE PRECISION CHECK (wer_reference_words {_FINITE}),
      latest_run_at TIMESTAMPTZ NOT NULL,
      CHECK ((wer_error_words IS NULL) = (wer_reference_words IS NULL)),
      CHECK (min_value <= p25 AND p25 <= p50 AND p50 <= p75 AND p75 <= p90
             AND p90 <= p95 AND p95 <= max_value),
      CHECK (grain = 'run' OR mod(extract(epoch FROM bucket_at)::bigint,
                                    CASE grain WHEN '1h' THEN 3600 ELSE 14400 END) = 0),
      PRIMARY KEY (provider, model, benchmark, dataset_id, metric_id, metric_version,
                   evaluation_variant, value_key, grain, bucket_at)
    );
    CREATE INDEX dashboard_rollups_lookup
      ON benchmarks_v2.dashboard_rollups (benchmark, dataset_id, grain, bucket_at);
    CREATE INDEX dashboard_rollups_slot
      ON benchmarks_v2.dashboard_rollups (grain, bucket_at);

    CREATE TABLE benchmarks_v2.dashboard_rollup_queue (
      slot_at TIMESTAMPTZ PRIMARY KEY,
      queued_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    INSERT INTO benchmarks_v2.dashboard_rollup_queue (slot_at)
    SELECT DISTINCT scheduled_at FROM benchmarks_v2.runs
    WHERE scheduled_at >= now() - interval '30 days' AND status IN ('succeeded', 'partial');

    DROP TABLE benchmarks_v2.dashboard_hourly_state;
    DROP TABLE benchmarks_v2.dashboard_source_refreshes;
    DROP TABLE benchmarks_v2.dashboard_hourly_aggregates;
    DROP TABLE benchmarks_v2.metric_values_by_bucket;
    ALTER TABLE benchmarks_v2.dashboard_summary_state RENAME TO dashboard_window_state;

    DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api') THEN
      GRANT SELECT ON benchmarks_v2.dashboard_rollups,
                      benchmarks_v2.dashboard_rollup_queue TO api;
    END IF; END $$;
    """)


def downgrade() -> None:
    op.execute("""
    ALTER TABLE benchmarks_v2.dashboard_window_state RENAME TO dashboard_summary_state;
    DROP TABLE benchmarks_v2.dashboard_rollup_queue;
    DROP TABLE benchmarks_v2.dashboard_rollups;

    CREATE TABLE benchmarks_v2.metric_values_by_bucket (
      provider TEXT NOT NULL CHECK (provider <> ''), model TEXT NOT NULL CHECK (model <> ''),
      benchmark TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S', 'LLM')),
      dataset_id TEXT NOT NULL CHECK (dataset_id <> ''),
      metric_id BIGINT NOT NULL REFERENCES benchmarks_v2.metrics(id) ON DELETE RESTRICT,
      metric_type TEXT NOT NULL CHECK (metric_type <> ''),
      metric_version TEXT NOT NULL CHECK (metric_version <> ''),
      evaluation_variant TEXT NOT NULL CHECK (evaluation_variant <> ''),
      value_key TEXT NOT NULL CHECK (value_key <> ''), unit TEXT NOT NULL CHECK (unit <> ''),
      bucket_at TIMESTAMPTZ NOT NULL,
      min_value DOUBLE PRECISION NOT NULL, p25 DOUBLE PRECISION NOT NULL,
      p50 DOUBLE PRECISION NOT NULL, p75 DOUBLE PRECISION NOT NULL,
      max_value DOUBLE PRECISION NOT NULL, value_sum DOUBLE PRECISION NOT NULL,
      sample_count INTEGER NOT NULL CHECK (sample_count > 0),
      CHECK (min_value <= p25 AND p25 <= p50 AND p50 <= p75 AND p75 <= max_value),
      PRIMARY KEY (provider, model, benchmark, dataset_id, metric_type, metric_version,
                   evaluation_variant, value_key, bucket_at)
    );
    CREATE INDEX metric_values_by_bucket_bucket_at ON benchmarks_v2.metric_values_by_bucket (bucket_at);
    CREATE INDEX metric_values_by_bucket_series_idx ON benchmarks_v2.metric_values_by_bucket
      (benchmark, dataset_id, metric_version, evaluation_variant, value_key, bucket_at);
    CREATE UNIQUE INDEX metric_values_by_bucket_metric_identity_key ON benchmarks_v2.metric_values_by_bucket
      (provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant, value_key, bucket_at);
    CREATE TRIGGER metric_values_by_bucket_sync_metric_identity
      BEFORE INSERT OR UPDATE ON benchmarks_v2.metric_values_by_bucket FOR EACH ROW
      EXECUTE FUNCTION benchmarks_v2.sync_normalized_metric_identity();

    CREATE TABLE benchmarks_v2.dashboard_hourly_aggregates (
      provider TEXT NOT NULL, model TEXT NOT NULL, benchmark TEXT NOT NULL, dataset_id TEXT NOT NULL,
      metric_id BIGINT NOT NULL,
      metric_type TEXT NOT NULL, metric_version TEXT NOT NULL, evaluation_variant TEXT NOT NULL,
      hour_at TIMESTAMPTZ NOT NULL CHECK (hour_at = date_trunc('hour', hour_at, 'UTC')),
      primary_sum DOUBLE PRECISION NOT NULL, sample_count BIGINT NOT NULL CHECK (sample_count > 0),
      numerator_sum DOUBLE PRECISION, denominator_sum DOUBLE PRECISION, coverage_complete BOOLEAN NOT NULL,
      source_count BIGINT NOT NULL CHECK (source_count > 0), latest_run_at TIMESTAMPTZ,
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
