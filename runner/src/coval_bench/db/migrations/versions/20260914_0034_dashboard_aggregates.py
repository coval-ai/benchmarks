# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Persist normalized dashboard summaries and hourly aggregate state."""

from __future__ import annotations

from alembic import op

revision = "20260914_0034"
down_revision = "20260911_0033"
branch_labels = None
depends_on = None

_VIEW_SQL = """
WITH evaluations AS (
 SELECT o.provider,o.model,o.benchmark,o.dataset_id,e.metric_type,e.metric_version,e.evaluation_variant,
        e.value,e.roundtrip,e.leading_silence,e.has_primary_role,e.wer_insertions_pct,
        e.wer_deletions_pct,e.wer_substitutions_pct,e.substitution_count,e.deletion_count,
        e.insertion_count,e.reference_words
 FROM benchmarks_v2.dashboard_metric_values e
 JOIN benchmarks_v2.benchmark_observations o ON o.id=e.observation_id
 JOIN benchmarks_v2.runs r ON r.id=o.run_id
 WHERE o.status='succeeded' AND r.status IN ('succeeded','partial')
   AND e.metric_version='v1' AND e.evaluation_variant='default'
   AND o.captured_at >= (SELECT as_of FROM benchmarks_v2.dashboard_summary_state WHERE id=true) - INTERVAL '__WINDOW__'
   AND o.captured_at < (SELECT as_of FROM benchmarks_v2.dashboard_summary_state WHERE id=true)
), public_values AS (
 SELECT e.* , p.metric_type AS public_metric, p.value AS public_value
 FROM evaluations e CROSS JOIN LATERAL (VALUES
   (e.metric_type,e.value), ('TTFARoundtrip',CASE WHEN e.metric_type='TTFA' THEN e.roundtrip END),
   ('TTFALeadingSilence',CASE WHEN e.metric_type='TTFA' THEN e.leading_silence END)
 ) p(metric_type,value) WHERE p.value IS NOT NULL
), grouped AS (
 SELECT provider,model,benchmark,
        CASE WHEN GROUPING(dataset_id)=1 THEN '__all__' ELSE dataset_id END dataset_id,
        public_metric metric_type,metric_version,evaluation_variant,
        AVG(public_value)::float8 mean_value,AVG(public_value)::float8 mean_for_ratio,
        COALESCE(STDDEV_SAMP(public_value),0)::float8 stddev_value,
        PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY public_value)::float8 p25,
        PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY public_value)::float8 p50,
        PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY public_value)::float8 p75,
        PERCENTILE_CONT(.9) WITHIN GROUP (ORDER BY public_value)::float8 p90,
        PERCENTILE_CONT(.95) WITHIN GROUP (ORDER BY public_value)::float8 p95,
        PERCENTILE_CONT(.99) WITHIN GROUP (ORDER BY public_value)::float8 p99,
        MIN(public_value)::float8 min_value,MAX(public_value)::float8 max_value,
        COUNT(*)::bigint sample_count,COUNT(*) FILTER (WHERE has_primary_role)::bigint primary_sample_count,
        CASE WHEN public_metric='WER' AND COUNT(reference_words)=COUNT(*) AND COUNT(substitution_count)=COUNT(*)
             AND COUNT(deletion_count)=COUNT(*) AND COUNT(insertion_count)=COUNT(*)
             THEN 100*SUM(substitution_count+deletion_count+insertion_count)/NULLIF(SUM(reference_words),0) END pooled_value,
        CASE WHEN public_metric='WER' AND COUNT(wer_insertions_pct)=COUNT(*)
             AND COUNT(wer_deletions_pct)=COUNT(*) AND COUNT(wer_substitutions_pct)=COUNT(*)
             THEN AVG(wer_insertions_pct) END wer_insertions_pct,
        CASE WHEN public_metric='WER' AND COUNT(wer_insertions_pct)=COUNT(*)
             AND COUNT(wer_deletions_pct)=COUNT(*) AND COUNT(wer_substitutions_pct)=COUNT(*)
             THEN AVG(wer_deletions_pct) END wer_deletions_pct,
        CASE WHEN public_metric='WER' AND COUNT(wer_insertions_pct)=COUNT(*)
             AND COUNT(wer_deletions_pct)=COUNT(*) AND COUNT(wer_substitutions_pct)=COUNT(*)
             THEN AVG(wer_substitutions_pct) END wer_substitutions_pct,
        CASE WHEN public_metric='WER' AND COUNT(reference_words)=COUNT(*) AND COUNT(substitution_count)=COUNT(*)
             AND COUNT(deletion_count)=COUNT(*) AND COUNT(insertion_count)=COUNT(*)
             THEN 100*SUM(insertion_count)/NULLIF(SUM(reference_words),0) END pooled_insertions_pct,
        CASE WHEN public_metric='WER' AND COUNT(reference_words)=COUNT(*) AND COUNT(substitution_count)=COUNT(*)
             AND COUNT(deletion_count)=COUNT(*) AND COUNT(insertion_count)=COUNT(*)
             THEN 100*SUM(deletion_count)/NULLIF(SUM(reference_words),0) END pooled_deletions_pct,
        CASE WHEN public_metric='WER' AND COUNT(reference_words)=COUNT(*) AND COUNT(substitution_count)=COUNT(*)
             AND COUNT(deletion_count)=COUNT(*) AND COUNT(insertion_count)=COUNT(*)
             THEN 100*SUM(substitution_count)/NULLIF(SUM(reference_words),0) END pooled_substitutions_pct
 FROM public_values GROUP BY GROUPING SETS
 ((provider,model,benchmark,dataset_id,public_metric,metric_version,evaluation_variant),
  (provider,model,benchmark,public_metric,metric_version,evaluation_variant))
)
SELECT provider,model,benchmark,dataset_id,
       benchmarks_v2.metric_id_for_code(metric_type) AS metric_id,metric_version,evaluation_variant,
       mean_value,COALESCE(pooled_value,mean_for_ratio) avg_value,stddev_value,p25,p50,p75,p90,p95,p99,
       min_value,max_value,sample_count,primary_sample_count,
       COALESCE(pooled_insertions_pct,wer_insertions_pct) AS wer_insertions_pct,
       COALESCE(pooled_deletions_pct,wer_deletions_pct) AS wer_deletions_pct,
       COALESCE(pooled_substitutions_pct,wer_substitutions_pct) AS wer_substitutions_pct,
       pooled_value,pooled_insertions_pct,pooled_deletions_pct,
       pooled_substitutions_pct,'{"schema_version" : 1}'::jsonb metadata FROM grouped
"""


def upgrade() -> None:
    op.execute("""
    CREATE TABLE benchmarks_v2.metrics (
      id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
      code TEXT NOT NULL UNIQUE CHECK (code <> ''),
      display_name TEXT NOT NULL CHECK (display_name <> ''));
    INSERT INTO benchmarks_v2.metrics (code,display_name) VALUES
      ('WER','Word Error Rate'),
      ('TTFT','Time to First Token'),
      ('TTFS','Time to Final from Speech'),
      ('TTFA','Time to First Audio'),
      ('TTFARoundtrip','TTFA Network Roundtrip'),
      ('TTFALeadingSilence','TTFA Leading Silence'),
      ('RTF','Real-Time Factor'),
      ('AudioToFinal','Audio to Final'),
      ('V2V','Voice-to-Voice Latency'),
      ('InstructionFollowing','Instruction Adherence'),
      ('InterruptionRate','Interruption Rate'),
      ('ExpectedBehaviorAdherence','Expected Behavior Adherence');

    CREATE FUNCTION benchmarks_v2.preserve_metric_identity() RETURNS trigger
    LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP IN ('DELETE','TRUNCATE') THEN
        RAISE EXCEPTION 'metric definitions must be retained' USING ERRCODE='23514';
      END IF;
      IF NEW.id IS DISTINCT FROM OLD.id OR NEW.code IS DISTINCT FROM OLD.code THEN
        RAISE EXCEPTION 'metric id and code are immutable' USING ERRCODE='23514';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER metrics_preserve_identity BEFORE UPDATE OR DELETE
      ON benchmarks_v2.metrics FOR EACH ROW
      EXECUTE FUNCTION benchmarks_v2.preserve_metric_identity();
    CREATE TRIGGER metrics_preserve_definitions BEFORE TRUNCATE
      ON benchmarks_v2.metrics FOR EACH STATEMENT
      EXECUTE FUNCTION benchmarks_v2.preserve_metric_identity();

    CREATE FUNCTION benchmarks_v2.metric_id_for_code(metric_code TEXT) RETURNS BIGINT
    LANGUAGE plpgsql STABLE AS $$
    DECLARE resolved_id BIGINT;
    BEGIN
      SELECT id INTO resolved_id FROM benchmarks_v2.metrics WHERE code=metric_code;
      IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown metric definition: %', metric_code USING ERRCODE='23503';
      END IF;
      RETURN resolved_id;
    END $$;
    """)
    op.execute("""
    CREATE TABLE benchmarks_v2.dashboard_summary_state (
      id BOOLEAN PRIMARY KEY CHECK (id), generation BIGINT NOT NULL DEFAULT 0,
      as_of TIMESTAMPTZ, published_at TIMESTAMPTZ, definition_revision INTEGER NOT NULL,
      definition_fingerprint TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{"schema_version" : 1}'::jsonb);
    INSERT INTO benchmarks_v2.dashboard_summary_state (id,definition_revision,definition_fingerprint)
      VALUES (true,2,'uninitialized');
    CREATE TABLE benchmarks_v2.dashboard_hourly_aggregates (
      provider TEXT NOT NULL, model TEXT NOT NULL, benchmark TEXT NOT NULL, dataset_id TEXT NOT NULL,
      metric_id BIGINT NOT NULL REFERENCES benchmarks_v2.metrics(id) ON DELETE RESTRICT,
      metric_version TEXT NOT NULL, evaluation_variant TEXT NOT NULL,
      hour_at TIMESTAMPTZ NOT NULL, primary_sum DOUBLE PRECISION NOT NULL, sample_count BIGINT NOT NULL,
      numerator_sum DOUBLE PRECISION, denominator_sum DOUBLE PRECISION, coverage_complete BOOLEAN NOT NULL,
      source_count BIGINT NOT NULL, latest_source_at TIMESTAMPTZ, definition_revision INTEGER NOT NULL,
      metadata JSONB NOT NULL DEFAULT '{"schema_version" : 1}'::jsonb,
      PRIMARY KEY(provider,model,benchmark,dataset_id,metric_id,metric_version,evaluation_variant,hour_at));
    CREATE TABLE benchmarks_v2.dashboard_hourly_state (
      hour_at TIMESTAMPTZ PRIMARY KEY, dirty BOOLEAN NOT NULL, refreshed_at TIMESTAMPTZ,
      definition_revision INTEGER NOT NULL, definition_fingerprint TEXT NOT NULL,
      metadata JSONB NOT NULL DEFAULT '{"schema_version" : 1}'::jsonb);
    CREATE TABLE benchmarks_v2.dashboard_source_refreshes (
      bucket_at TIMESTAMPTZ PRIMARY KEY,
      requested_at TIMESTAMPTZ NOT NULL DEFAULT now());
    """)
    op.execute("""
      ALTER TABLE benchmarks_v2.dashboard_summary_state
        ADD CHECK (generation >= 0), ADD CHECK (jsonb_typeof(metadata)='object');
      ALTER TABLE benchmarks_v2.dashboard_hourly_aggregates
        ADD CHECK (hour_at=date_trunc('hour',hour_at,'UTC')),
        ADD CHECK (sample_count > 0), ADD CHECK (source_count > 0),
        ADD CHECK (jsonb_typeof(metadata)='object');
      ALTER TABLE benchmarks_v2.dashboard_hourly_state
        ADD CHECK (hour_at=date_trunc('hour',hour_at,'UTC')),
        ADD CHECK (jsonb_typeof(metadata)='object');
    """)
    for name, window in (("24h", "24 hours"), ("7d", "7 days"), ("30d", "30 days")):
        op.execute(
            f"CREATE MATERIALIZED VIEW benchmarks_v2.normalized_results_{name} AS "
            + _VIEW_SQL.replace("__WINDOW__", window)
            + " WITH NO DATA"
        )
        op.execute(
            f"CREATE UNIQUE INDEX normalized_results_{name}_key ON benchmarks_v2.normalized_results_{name} (provider,model,benchmark,dataset_id,metric_id,metric_version,evaluation_variant)"
        )
        op.execute(
            f"CREATE INDEX normalized_results_{name}_lookup ON benchmarks_v2.normalized_results_{name} (benchmark,dataset_id,metric_id,metric_version,evaluation_variant)"
        )
    op.execute(
        "CREATE INDEX dashboard_hourly_aggregates_hour_idx ON benchmarks_v2.dashboard_hourly_aggregates (hour_at)"
    )
    op.execute(
        "CREATE INDEX dashboard_hourly_aggregates_lookup ON benchmarks_v2.dashboard_hourly_aggregates (benchmark,dataset_id,hour_at)"
    )
    op.execute("""
      DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='api') THEN
        GRANT SELECT ON benchmarks_v2.metrics,
          benchmarks_v2.dashboard_summary_state, benchmarks_v2.dashboard_hourly_aggregates,
          benchmarks_v2.dashboard_hourly_state, benchmarks_v2.dashboard_source_refreshes,
          benchmarks_v2.normalized_results_24h,
          benchmarks_v2.normalized_results_7d, benchmarks_v2.normalized_results_30d TO api;
      END IF; END $$;
    """)


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.normalized_results_30d")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.normalized_results_7d")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.normalized_results_24h")
    op.execute(
        "DROP TABLE IF EXISTS benchmarks_v2.dashboard_source_refreshes, benchmarks_v2.dashboard_hourly_aggregates, benchmarks_v2.dashboard_hourly_state, benchmarks_v2.dashboard_summary_state"
    )
    op.execute("DROP FUNCTION benchmarks_v2.metric_id_for_code(TEXT)")
    op.execute("DROP TABLE benchmarks_v2.metrics")
    op.execute("DROP FUNCTION benchmarks_v2.preserve_metric_identity()")
