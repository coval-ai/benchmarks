# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Add normalized, immutable benchmark observation and metric storage."""

from __future__ import annotations

from alembic import op

revision = "20260813_0018"
down_revision = "20260812_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create additive normalized tables; legacy result storage remains untouched."""
    op.execute(
        """
        CREATE TABLE benchmarks_v2.benchmark_observations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id BIGINT NOT NULL REFERENCES benchmarks_v2.runs(id) ON DELETE CASCADE,
            dataset_id TEXT NOT NULL CHECK (dataset_id <> ''),
            dataset_sha256 TEXT NOT NULL CHECK (dataset_sha256 ~ '^[0-9a-f]{64}$'),
            sample_id TEXT NOT NULL CHECK (sample_id <> ''),
            provider TEXT NOT NULL CHECK (provider <> ''),
            model TEXT NOT NULL CHECK (model <> ''),
            voice TEXT,
            benchmark TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S')),
            -- source_kind is only a category; dataset_id and dataset_sha256 identify the dataset.
            source_kind TEXT NOT NULL CHECK (source_kind IN ('dataset_audio', 'generated_audio', 'conversation_audio')),
            transport_protocol TEXT,
            submit_to_headers_ms DOUBLE PRECISION CHECK (submit_to_headers_ms >= 0 AND submit_to_headers_ms NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)),
            provider_extras JSONB CHECK (provider_extras IS NULL OR jsonb_typeof(provider_extras) = 'object'),
            captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
            error TEXT,
            failure_origin TEXT CHECK (failure_origin IN ('provider', 'runner')),
            CHECK ((status = 'succeeded' AND error IS NULL AND failure_origin IS NULL) OR (status = 'failed' AND error IS NOT NULL AND error <> '' AND failure_origin IS NOT NULL)),
            UNIQUE NULLS NOT DISTINCT (run_id, sample_id, provider, model, voice)
        );

        CREATE TABLE benchmarks_v2.observation_artifacts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            observation_id UUID NOT NULL REFERENCES benchmarks_v2.benchmark_observations(id) ON DELETE CASCADE,
            artifact_type TEXT NOT NULL CHECK (artifact_type IN ('provider_transcript', 'generated_audio', 'conversation_audio', 'conversation_trace', 'timing_events')),
            schema_name TEXT NOT NULL CHECK (schema_name <> ''),
            schema_version TEXT NOT NULL CHECK (schema_version <> ''),
            gcs_uri TEXT NOT NULL CHECK (gcs_uri ~ '^gs://[^/?#]+/[^/?#][^?#]*$'),
            content_sha256 TEXT NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
            size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
            duration_ms DOUBLE PRECISION CHECK (duration_ms IS NULL OR (duration_ms > 0 AND duration_ms NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8))),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (observation_id, artifact_type)
        );

        CREATE TABLE benchmarks_v2.preprocessing_artifacts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            observation_id UUID NOT NULL REFERENCES benchmarks_v2.benchmark_observations(id) ON DELETE CASCADE,
            pipeline TEXT NOT NULL CHECK (pipeline <> ''),
            pipeline_version TEXT NOT NULL CHECK (pipeline_version <> ''),
            artifact_name TEXT NOT NULL CHECK (artifact_name <> ''),
            schema_name TEXT NOT NULL CHECK (schema_name <> ''),
            schema_version TEXT NOT NULL CHECK (schema_version <> ''),
            producer_name TEXT NOT NULL CHECK (producer_name <> ''),
            producer_provider TEXT NOT NULL CHECK (producer_provider <> ''),
            producer_model TEXT NOT NULL CHECK (producer_model <> ''),
            producer_version TEXT NOT NULL CHECK (producer_version <> ''),
            gcs_uri TEXT NOT NULL CHECK (gcs_uri ~ '^gs://[^/?#]+/[^/?#][^?#]*$'),
            content_sha256 TEXT NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (observation_id, pipeline, pipeline_version, artifact_name, schema_name, schema_version,
                    producer_name, producer_provider, producer_model, producer_version)
        );

        CREATE TABLE benchmarks_v2.metric_evaluations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            observation_id UUID NOT NULL REFERENCES benchmarks_v2.benchmark_observations(id) ON DELETE CASCADE,
            metric_type TEXT NOT NULL CHECK (metric_type <> ''), metric_version TEXT NOT NULL CHECK (metric_version <> ''),
            evaluation_variant TEXT NOT NULL DEFAULT 'default' CHECK (evaluation_variant <> ''),
            executor TEXT NOT NULL CHECK (executor <> ''), external_request_id TEXT,
            status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
            started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, error TEXT CHECK (error IS NULL OR error <> ''),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (updated_at >= created_at), CHECK (finished_at IS NULL OR (started_at IS NOT NULL AND finished_at >= started_at)),
            CHECK ((status = 'queued' AND started_at IS NULL AND finished_at IS NULL AND error IS NULL) OR (status = 'running' AND started_at IS NOT NULL AND finished_at IS NULL AND error IS NULL) OR (status = 'succeeded' AND started_at IS NOT NULL AND finished_at IS NOT NULL AND error IS NULL) OR (status = 'failed' AND started_at IS NOT NULL AND finished_at IS NOT NULL AND error IS NOT NULL)),
            UNIQUE (observation_id, metric_type, metric_version, evaluation_variant)
        );
        CREATE TABLE benchmarks_v2.metric_evaluation_inputs (
            metric_evaluation_id UUID NOT NULL REFERENCES benchmarks_v2.metric_evaluations(id) ON DELETE CASCADE,
            observation_artifact_id UUID REFERENCES benchmarks_v2.observation_artifacts(id) ON DELETE CASCADE,
            preprocessing_artifact_id UUID REFERENCES benchmarks_v2.preprocessing_artifacts(id) ON DELETE CASCADE,
            input_role TEXT NOT NULL CHECK (input_role <> ''),
            input_order INTEGER NOT NULL CHECK (input_order >= 0),
            PRIMARY KEY (metric_evaluation_id, input_role, input_order),
            CHECK (num_nonnulls(observation_artifact_id, preprocessing_artifact_id) = 1)
        );
        CREATE UNIQUE INDEX metric_evaluation_inputs_observation_artifact_unique ON benchmarks_v2.metric_evaluation_inputs (metric_evaluation_id, observation_artifact_id) WHERE observation_artifact_id IS NOT NULL;
        CREATE UNIQUE INDEX metric_evaluation_inputs_preprocessing_artifact_unique ON benchmarks_v2.metric_evaluation_inputs (metric_evaluation_id, preprocessing_artifact_id) WHERE preprocessing_artifact_id IS NOT NULL;
        CREATE INDEX metric_evaluation_inputs_observation_artifact_id ON benchmarks_v2.metric_evaluation_inputs (observation_artifact_id) WHERE observation_artifact_id IS NOT NULL;
        CREATE INDEX metric_evaluation_inputs_artifact_id ON benchmarks_v2.metric_evaluation_inputs (preprocessing_artifact_id) WHERE preprocessing_artifact_id IS NOT NULL;
        CREATE TABLE benchmarks_v2.metric_values (
            metric_evaluation_id UUID NOT NULL REFERENCES benchmarks_v2.metric_evaluations(id) ON DELETE CASCADE,
            value_key TEXT NOT NULL CHECK (value_key <> ''), unit TEXT NOT NULL CHECK (unit <> ''),
            value DOUBLE PRECISION NOT NULL CHECK (value NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)),
            is_primary BOOLEAN NOT NULL DEFAULT false, PRIMARY KEY (metric_evaluation_id, value_key)
        );
        CREATE UNIQUE INDEX metric_values_one_primary ON benchmarks_v2.metric_values (metric_evaluation_id) WHERE is_primary;
        CREATE TABLE benchmarks_v2.metric_artifacts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            metric_evaluation_id UUID NOT NULL REFERENCES benchmarks_v2.metric_evaluations(id) ON DELETE CASCADE,
            artifact_type TEXT NOT NULL CHECK (artifact_type <> ''), uri TEXT NOT NULL CHECK (uri ~ '^gs://[^/?#]+/[^/?#][^?#]*$'),
            sha256 TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'), size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE (metric_evaluation_id, artifact_type, sha256)
        );
        CREATE TABLE benchmarks_v2.metric_values_by_bucket (
            provider TEXT NOT NULL CHECK (provider <> ''), model TEXT NOT NULL CHECK (model <> ''),
            benchmark TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S')), dataset_id TEXT NOT NULL CHECK (dataset_id <> ''),
            metric_type TEXT NOT NULL CHECK (metric_type <> ''), metric_version TEXT NOT NULL CHECK (metric_version <> ''), evaluation_variant TEXT NOT NULL CHECK (evaluation_variant <> ''),
            value_key TEXT NOT NULL CHECK (value_key <> ''), unit TEXT NOT NULL CHECK (unit <> ''), bucket_at TIMESTAMPTZ NOT NULL,
            min_value DOUBLE PRECISION NOT NULL CHECK (min_value NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), p25 DOUBLE PRECISION NOT NULL CHECK (p25 NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), p50 DOUBLE PRECISION NOT NULL CHECK (p50 NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), p75 DOUBLE PRECISION NOT NULL CHECK (p75 NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), max_value DOUBLE PRECISION NOT NULL CHECK (max_value NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), value_sum DOUBLE PRECISION NOT NULL CHECK (value_sum NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), sample_count INTEGER NOT NULL CHECK (sample_count > 0),
            CHECK (min_value <= p25 AND p25 <= p50 AND p50 <= p75 AND p75 <= max_value),
            PRIMARY KEY (provider, model, benchmark, dataset_id, metric_type, metric_version, evaluation_variant, value_key, bucket_at)
        );
        CREATE INDEX metric_values_by_bucket_bucket_at ON benchmarks_v2.metric_values_by_bucket (bucket_at);

        -- Lifecycle writes live in RunWriter's metric evaluation methods.
        CREATE FUNCTION benchmarks_v2.validate_metric_transition() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN IF NEW.status <> 'queued' THEN RAISE EXCEPTION 'work rows must be created queued'; END IF; RETURN NEW; END IF;
            IF TG_OP = 'DELETE' THEN
                IF NOT EXISTS (SELECT 1 FROM benchmarks_v2.benchmark_observations WHERE id = OLD.observation_id) THEN RETURN OLD; END IF;
                IF OLD.status IN ('succeeded', 'failed') THEN RAISE EXCEPTION 'terminal work rows are immutable'; END IF;
                RETURN OLD;
            END IF;
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.observation_id IS DISTINCT FROM OLD.observation_id
               OR NEW.metric_type IS DISTINCT FROM OLD.metric_type
               OR NEW.metric_version IS DISTINCT FROM OLD.metric_version
               OR NEW.evaluation_variant IS DISTINCT FROM OLD.evaluation_variant
               OR NEW.executor IS DISTINCT FROM OLD.executor
               OR NEW.external_request_id IS DISTINCT FROM OLD.external_request_id
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION 'metric evaluation identity is immutable';
            END IF;
            IF OLD.status IN ('succeeded', 'failed') THEN RAISE EXCEPTION 'terminal work rows are immutable'; END IF;
            IF NOT ((OLD.status = 'queued' AND NEW.status IN ('running', 'failed')) OR (OLD.status = 'running' AND NEW.status IN ('succeeded', 'failed'))) THEN RAISE EXCEPTION 'invalid work status transition from % to %', OLD.status, NEW.status; END IF;
            RETURN NEW;
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER metric_evaluations_validate_transition BEFORE INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_evaluations FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.validate_metric_transition();
        CREATE FUNCTION benchmarks_v2.guard_immutable_preprocessing_artifact() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'DELETE' AND NOT EXISTS (SELECT 1 FROM benchmarks_v2.benchmark_observations WHERE id = OLD.observation_id) THEN RETURN OLD; END IF;
            RAISE EXCEPTION 'preprocessing artifacts are immutable';
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER preprocessing_artifacts_immutable BEFORE UPDATE OR DELETE ON benchmarks_v2.preprocessing_artifacts FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_immutable_preprocessing_artifact();
        CREATE FUNCTION benchmarks_v2.guard_immutable_observation_artifact() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'DELETE' AND NOT EXISTS (SELECT 1 FROM benchmarks_v2.benchmark_observations WHERE id = OLD.observation_id) THEN RETURN OLD; END IF;
            RAISE EXCEPTION 'observation artifacts are immutable';
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER observation_artifacts_immutable BEFORE UPDATE OR DELETE ON benchmarks_v2.observation_artifacts FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_immutable_observation_artifact();
        CREATE FUNCTION benchmarks_v2.guard_metric_evaluation_input() RETURNS trigger AS $$
        DECLARE evaluation_observation UUID; artifact_observation UUID; evaluation_status TEXT;
        BEGIN
            IF TG_OP = 'DELETE' THEN
                IF NOT EXISTS (SELECT 1 FROM benchmarks_v2.metric_evaluations WHERE id = OLD.metric_evaluation_id)
                   OR (OLD.observation_artifact_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM benchmarks_v2.observation_artifacts WHERE id = OLD.observation_artifact_id))
                   OR (OLD.preprocessing_artifact_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM benchmarks_v2.preprocessing_artifacts WHERE id = OLD.preprocessing_artifact_id)) THEN RETURN OLD; END IF;
                RAISE EXCEPTION 'metric evaluation inputs are immutable';
            END IF;
            IF TG_OP = 'UPDATE' THEN RAISE EXCEPTION 'metric evaluation inputs are immutable'; END IF;
            SELECT observation_id, status INTO evaluation_observation, evaluation_status
              FROM benchmarks_v2.metric_evaluations
              WHERE id = NEW.metric_evaluation_id FOR UPDATE;
            IF NEW.observation_artifact_id IS NOT NULL THEN
                SELECT observation_id INTO artifact_observation FROM benchmarks_v2.observation_artifacts WHERE id = NEW.observation_artifact_id;
            ELSE
                SELECT observation_id INTO artifact_observation FROM benchmarks_v2.preprocessing_artifacts WHERE id = NEW.preprocessing_artifact_id;
            END IF;
            IF evaluation_status IS DISTINCT FROM 'queued' THEN RAISE EXCEPTION 'metric evaluation inputs can only be inserted while queued'; END IF;
            IF evaluation_observation IS DISTINCT FROM artifact_observation THEN RAISE EXCEPTION 'metric evaluation input must share the evaluation observation'; END IF;
            RETURN NEW;
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER metric_evaluation_inputs_guard BEFORE INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_evaluation_inputs FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_metric_evaluation_input();
        CREATE FUNCTION benchmarks_v2.guard_terminal_metric_payload() RETURNS trigger AS $$
        DECLARE parent_status TEXT;
        BEGIN
            SELECT status INTO parent_status FROM benchmarks_v2.metric_evaluations WHERE id = CASE WHEN TG_OP = 'DELETE' THEN OLD.metric_evaluation_id ELSE NEW.metric_evaluation_id END;
            IF FOUND AND parent_status IN ('succeeded', 'failed') THEN RAISE EXCEPTION 'terminal work payloads are immutable'; END IF;
            RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER metric_values_guard_terminal BEFORE INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_values FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_terminal_metric_payload();
        CREATE TRIGGER metric_artifacts_guard_terminal BEFORE INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_artifacts FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_terminal_metric_payload();
        CREATE FUNCTION benchmarks_v2.validate_metric_evaluation_success_outputs() RETURNS trigger AS $$
        DECLARE target_id UUID; target_status TEXT; value_count INTEGER; primary_count INTEGER; artifact_count INTEGER;
        BEGIN
            IF TG_TABLE_NAME = 'metric_evaluations' THEN
                target_id := CASE WHEN TG_OP = 'DELETE' THEN OLD.id ELSE NEW.id END;
            ELSE
                target_id := CASE WHEN TG_OP = 'DELETE' THEN OLD.metric_evaluation_id ELSE NEW.metric_evaluation_id END;
            END IF;
            SELECT status INTO target_status FROM benchmarks_v2.metric_evaluations WHERE id = target_id; IF NOT FOUND THEN RETURN NULL; END IF;
            SELECT count(*), count(*) FILTER (WHERE is_primary) INTO value_count, primary_count FROM benchmarks_v2.metric_values WHERE metric_evaluation_id = target_id;
            SELECT count(*) INTO artifact_count FROM benchmarks_v2.metric_artifacts WHERE metric_evaluation_id = target_id;
            IF target_status <> 'succeeded' THEN IF value_count <> 0 OR artifact_count <> 0 THEN RAISE EXCEPTION 'metric outputs require a succeeded evaluation'; END IF; RETURN NULL; END IF;
            IF value_count = 0 OR primary_count <> 1 THEN RAISE EXCEPTION 'succeeded metric evaluation requires values and exactly one primary'; END IF;
            RETURN NULL;
        END; $$ LANGUAGE plpgsql;
        CREATE CONSTRAINT TRIGGER metric_evaluations_validate_success_outputs AFTER INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_evaluations DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.validate_metric_evaluation_success_outputs();
        CREATE CONSTRAINT TRIGGER metric_values_validate_success AFTER INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_values DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.validate_metric_evaluation_success_outputs();
        CREATE CONSTRAINT TRIGGER metric_artifacts_validate_success AFTER INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_artifacts DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.validate_metric_evaluation_success_outputs();
        DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api') THEN EXECUTE 'REVOKE ALL PRIVILEGES ON TABLE benchmarks_v2.benchmark_observations, benchmarks_v2.observation_artifacts, benchmarks_v2.preprocessing_artifacts, benchmarks_v2.metric_evaluations, benchmarks_v2.metric_evaluation_inputs, benchmarks_v2.metric_values, benchmarks_v2.metric_artifacts, benchmarks_v2.metric_values_by_bucket FROM api'; END IF; END; $$;
        """
    )


def downgrade() -> None:
    """Remove only the additive normalized storage objects."""
    op.execute(
        """
        DROP TRIGGER IF EXISTS metric_artifacts_validate_success ON benchmarks_v2.metric_artifacts;
        DROP TRIGGER IF EXISTS metric_values_validate_success ON benchmarks_v2.metric_values;
        DROP TRIGGER IF EXISTS metric_evaluations_validate_success_outputs ON benchmarks_v2.metric_evaluations;
        DROP FUNCTION IF EXISTS benchmarks_v2.validate_metric_evaluation_success_outputs();
        DROP TRIGGER IF EXISTS metric_artifacts_guard_terminal ON benchmarks_v2.metric_artifacts;
        DROP TRIGGER IF EXISTS metric_values_guard_terminal ON benchmarks_v2.metric_values;
        DROP FUNCTION IF EXISTS benchmarks_v2.guard_terminal_metric_payload();
        DROP TRIGGER IF EXISTS observation_artifacts_immutable ON benchmarks_v2.observation_artifacts;
        DROP FUNCTION IF EXISTS benchmarks_v2.guard_immutable_observation_artifact();
        DROP TRIGGER IF EXISTS preprocessing_artifacts_immutable ON benchmarks_v2.preprocessing_artifacts;
        DROP FUNCTION IF EXISTS benchmarks_v2.guard_immutable_preprocessing_artifact();
        DROP TRIGGER IF EXISTS metric_evaluation_inputs_guard ON benchmarks_v2.metric_evaluation_inputs;
        DROP FUNCTION IF EXISTS benchmarks_v2.guard_metric_evaluation_input();
        DROP TRIGGER IF EXISTS metric_evaluations_validate_transition ON benchmarks_v2.metric_evaluations;
        DROP FUNCTION IF EXISTS benchmarks_v2.validate_metric_transition();
        DROP TABLE IF EXISTS benchmarks_v2.metric_values_by_bucket;
        DROP TABLE IF EXISTS benchmarks_v2.metric_artifacts;
        DROP TABLE IF EXISTS benchmarks_v2.metric_values;
        DROP TABLE IF EXISTS benchmarks_v2.metric_evaluation_inputs;
        DROP TABLE IF EXISTS benchmarks_v2.metric_evaluations;
        DROP TABLE IF EXISTS benchmarks_v2.preprocessing_artifacts;
        DROP TABLE IF EXISTS benchmarks_v2.observation_artifacts;
        DROP TABLE IF EXISTS benchmarks_v2.benchmark_observations;
        """
    )
