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
            source_kind TEXT NOT NULL CHECK (source_kind IN ('dataset_audio', 'generated_audio', 'conversation_audio')),
            audio_filename TEXT,
            transport_protocol TEXT,
            submit_to_headers_ms DOUBLE PRECISION CHECK (submit_to_headers_ms IS NULL OR (submit_to_headers_ms NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8) AND submit_to_headers_ms >= 0)),
            audio_uri TEXT CHECK (audio_uri IS NULL OR audio_uri ~ '^gs://[^/?#]+/[^/?#][^?#]*$'),
            audio_sha256 TEXT CHECK (audio_sha256 IS NULL OR audio_sha256 ~ '^[0-9a-f]{64}$'),
            audio_size_bytes BIGINT CHECK (audio_size_bytes IS NULL OR audio_size_bytes > 0),
            audio_duration_ms INTEGER CHECK (audio_duration_ms IS NULL OR audio_duration_ms > 0),
            captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
            error TEXT,
            CHECK ((audio_uri IS NULL AND audio_sha256 IS NULL AND audio_size_bytes IS NULL AND audio_duration_ms IS NULL) OR (audio_uri IS NOT NULL AND audio_sha256 IS NOT NULL AND audio_size_bytes IS NOT NULL AND audio_duration_ms IS NOT NULL)),
            CHECK ((status = 'succeeded' AND error IS NULL) OR (status = 'failed' AND error IS NOT NULL AND error <> '')),
            UNIQUE NULLS NOT DISTINCT (run_id, dataset_id, sample_id, provider, model, voice, benchmark)
        );

        CREATE TABLE benchmarks_v2.preprocessing_artifacts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            observation_id UUID NOT NULL REFERENCES benchmarks_v2.benchmark_observations(id) ON DELETE CASCADE,
            pipeline TEXT NOT NULL CHECK (pipeline <> ''),
            pipeline_version TEXT NOT NULL CHECK (pipeline_version <> ''),
            artifact_name TEXT NOT NULL CHECK (artifact_name IN ('word_timestamps', 'phoneme_timestamps')),
            schema_name TEXT NOT NULL CHECK (schema_name IN ('WordTimestampsV1', 'PhonemeTimestampsV1')),
            schema_version TEXT NOT NULL CHECK (schema_version = 'v1'),
            producer_name TEXT NOT NULL CHECK (producer_name IN ('word_aligner', 'phoneme_aligner')),
            producer_version TEXT NOT NULL CHECK (producer_version <> ''),
            gcs_uri TEXT NOT NULL CHECK (gcs_uri ~ '^gs://[^/?#]+/[^/?#][^?#]*$'),
            content_sha256 TEXT NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
            size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
            duration_ms DOUBLE PRECISION CHECK (duration_ms IS NULL OR (duration_ms NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8) AND duration_ms > 0)),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK ((artifact_name = 'word_timestamps' AND schema_name = 'WordTimestampsV1' AND producer_name = 'word_aligner') OR (artifact_name = 'phoneme_timestamps' AND schema_name = 'PhonemeTimestampsV1' AND producer_name = 'phoneme_aligner')),
            UNIQUE (observation_id, pipeline, pipeline_version, artifact_name, producer_version)
        );

        CREATE TABLE benchmarks_v2.metric_evaluations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            observation_id UUID NOT NULL REFERENCES benchmarks_v2.benchmark_observations(id) ON DELETE CASCADE,
            metric_type TEXT NOT NULL, metric_version TEXT NOT NULL,
            CHECK ((metric_type, metric_version) IN (
                ('WER', 'v1'), ('TTFT', 'v1'), ('TTFS', 'v1'), ('TTFA', 'v1'),
                ('TTFARoundtrip', 'v1'), ('TTFALeadingSilence', 'v1'), ('RTF', 'v1'),
                ('AudioToFinal', 'v1'), ('V2V', 'v1'), ('InstructionFollowing', 'v1'),
                ('InterruptionRate', 'v1')
            )),
            executor TEXT NOT NULL CHECK (executor IN ('inline', 'coval_api')), external_request_id TEXT,
            status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'partial', 'succeeded', 'failed')),
            started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, error TEXT CHECK (error IS NULL OR error <> ''),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (updated_at >= created_at), CHECK (finished_at IS NULL OR (started_at IS NOT NULL AND finished_at >= started_at)),
            CHECK ((status = 'queued' AND started_at IS NULL AND finished_at IS NULL AND error IS NULL) OR (status = 'running' AND started_at IS NOT NULL AND finished_at IS NULL AND error IS NULL) OR (status = 'partial' AND started_at IS NOT NULL AND finished_at IS NOT NULL) OR (status = 'succeeded' AND started_at IS NOT NULL AND finished_at IS NOT NULL AND error IS NULL) OR (status = 'failed' AND started_at IS NOT NULL AND finished_at IS NOT NULL AND error IS NOT NULL)),
            UNIQUE (observation_id, metric_type, metric_version)
        );
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
            metric_type TEXT NOT NULL CHECK (metric_type <> ''), metric_version TEXT NOT NULL CHECK (metric_version <> ''),
            value_key TEXT NOT NULL CHECK (value_key <> ''), unit TEXT NOT NULL CHECK (unit <> ''), bucket_at TIMESTAMPTZ NOT NULL,
            min_value DOUBLE PRECISION NOT NULL CHECK (min_value NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), p25 DOUBLE PRECISION NOT NULL CHECK (p25 NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), p50 DOUBLE PRECISION NOT NULL CHECK (p50 NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), p75 DOUBLE PRECISION NOT NULL CHECK (p75 NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), max_value DOUBLE PRECISION NOT NULL CHECK (max_value NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), value_sum DOUBLE PRECISION NOT NULL CHECK (value_sum NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)), sample_count INTEGER NOT NULL CHECK (sample_count > 0),
            CHECK (min_value <= p25 AND p25 <= p50 AND p50 <= p75 AND p75 <= max_value),
            PRIMARY KEY (provider, model, benchmark, dataset_id, metric_type, metric_version, value_key, bucket_at)
        );

        -- Lifecycle writes live in RunWriter's metric evaluation methods.
        CREATE FUNCTION benchmarks_v2.validate_metric_transition() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN IF NEW.status <> 'queued' THEN RAISE EXCEPTION 'work rows must be created queued'; END IF; RETURN NEW; END IF;
            IF TG_OP = 'DELETE' THEN
                IF pg_trigger_depth() > 1 THEN RETURN OLD; END IF;
                IF OLD.status IN ('partial', 'succeeded', 'failed') THEN RAISE EXCEPTION 'terminal work rows are immutable'; END IF;
                RETURN OLD;
            END IF;
            IF OLD.status IN ('partial', 'succeeded', 'failed') THEN RAISE EXCEPTION 'terminal work rows are immutable'; END IF;
            IF NOT ((OLD.status = 'queued' AND NEW.status IN ('running', 'failed')) OR (OLD.status = 'running' AND NEW.status IN ('partial', 'succeeded', 'failed'))) THEN RAISE EXCEPTION 'invalid work status transition from % to %', OLD.status, NEW.status; END IF;
            RETURN NEW;
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER metric_evaluations_validate_transition BEFORE INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_evaluations FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.validate_metric_transition();
        CREATE FUNCTION benchmarks_v2.guard_immutable_preprocessing_artifact() RETURNS trigger AS $$
        BEGIN
            IF pg_trigger_depth() > 1 THEN RETURN OLD; END IF;
            RAISE EXCEPTION 'preprocessing artifacts are immutable';
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER preprocessing_artifacts_immutable BEFORE UPDATE OR DELETE ON benchmarks_v2.preprocessing_artifacts FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_immutable_preprocessing_artifact();
        CREATE FUNCTION benchmarks_v2.guard_terminal_metric_payload() RETURNS trigger AS $$
        DECLARE parent_status TEXT;
        BEGIN
            SELECT status INTO parent_status FROM benchmarks_v2.metric_evaluations WHERE id = CASE WHEN TG_OP = 'DELETE' THEN OLD.metric_evaluation_id ELSE NEW.metric_evaluation_id END;
            IF FOUND AND parent_status IN ('partial', 'succeeded', 'failed') THEN RAISE EXCEPTION 'terminal work payloads are immutable'; END IF;
            RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER metric_values_guard_terminal BEFORE INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_values FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_terminal_metric_payload();
        CREATE TRIGGER metric_artifacts_guard_terminal BEFORE INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_artifacts FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_terminal_metric_payload();
        CREATE FUNCTION benchmarks_v2.validate_metric_evaluation_success_outputs() RETURNS trigger AS $$
        DECLARE target_id UUID; metric_kind TEXT; metric_ver TEXT; target_status TEXT; value_keys TEXT[]; primary_value DOUBLE PRECISION; primary_unit TEXT; component_sum DOUBLE PRECISION; primary_count INTEGER; incorrectly_designated_count INTEGER; artifact_count INTEGER;
        BEGIN
            IF TG_TABLE_NAME = 'metric_evaluations' THEN
                target_id := CASE WHEN TG_OP = 'DELETE' THEN OLD.id ELSE NEW.id END;
            ELSE
                target_id := CASE WHEN TG_OP = 'DELETE' THEN OLD.metric_evaluation_id ELSE NEW.metric_evaluation_id END;
            END IF;
            SELECT metric_type, metric_version, status INTO metric_kind, metric_ver, target_status FROM benchmarks_v2.metric_evaluations WHERE id = target_id; IF NOT FOUND THEN RETURN NULL; END IF;
            SELECT array_agg(value_key ORDER BY value_key), max(value) FILTER (WHERE value_key = 'primary'), max(unit) FILTER (WHERE value_key = 'primary'), sum(value) FILTER (WHERE value_key <> 'primary'), count(*) FILTER (WHERE is_primary), count(*) FILTER (WHERE (value_key = 'primary') <> is_primary) INTO value_keys, primary_value, primary_unit, component_sum, primary_count, incorrectly_designated_count FROM benchmarks_v2.metric_values WHERE metric_evaluation_id = target_id;
            SELECT count(*) INTO artifact_count FROM benchmarks_v2.metric_artifacts WHERE metric_evaluation_id = target_id;
            IF target_status NOT IN ('partial', 'succeeded') THEN IF value_keys IS NOT NULL OR artifact_count <> 0 THEN RAISE EXCEPTION 'metric outputs require a partial or succeeded evaluation'; END IF; RETURN NULL; END IF;
            IF target_status = 'partial' THEN RETURN NULL; END IF;
            IF metric_kind = 'WER' THEN IF value_keys <> ARRAY['deletions', 'insertions', 'primary', 'substitutions']::TEXT[] THEN RAISE EXCEPTION 'succeeded WER evaluation requires primary and all components'; END IF;
            ELSIF metric_kind = 'TTFA' THEN IF value_keys <> ARRAY['primary']::TEXT[] AND value_keys <> ARRAY['leading_silence', 'primary', 'roundtrip']::TEXT[] THEN RAISE EXCEPTION 'succeeded TTFA evaluation requires primary and optional all-or-none components'; END IF;
            ELSIF value_keys <> ARRAY['primary']::TEXT[] THEN RAISE EXCEPTION 'succeeded metric evaluation requires exactly one primary value'; END IF;
            IF primary_count <> 1 OR incorrectly_designated_count <> 0 THEN RAISE EXCEPTION 'metric evaluation requires exactly one correctly designated primary value'; END IF;
            IF metric_ver <> 'v1' THEN RAISE EXCEPTION 'unsupported metric/version'; END IF;
            IF (metric_kind IN ('WER', 'InstructionFollowing') AND primary_unit <> 'percent')
                OR (metric_kind IN ('TTFT', 'TTFS', 'AudioToFinal') AND primary_unit <> 'seconds')
                OR (metric_kind IN ('TTFA', 'TTFARoundtrip', 'TTFALeadingSilence', 'V2V') AND primary_unit <> 'milliseconds')
                OR (metric_kind = 'RTF' AND primary_unit <> 'ratio')
                OR (metric_kind = 'InterruptionRate' AND primary_unit <> 'per_minute')
                OR primary_value < 0
                OR (metric_kind = 'InstructionFollowing' AND primary_value > 100) THEN
                RAISE EXCEPTION 'metric primary value violates its contract';
            END IF;
            IF component_sum IS NOT NULL AND abs(component_sum - primary_value) > (CASE WHEN metric_kind = 'WER' THEN 0.0001 ELSE 0.001 END) THEN RAISE EXCEPTION 'metric components must sum to primary within tolerance'; END IF;
            RETURN NULL;
        END; $$ LANGUAGE plpgsql;
        CREATE CONSTRAINT TRIGGER metric_evaluations_validate_success_outputs AFTER INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_evaluations DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.validate_metric_evaluation_success_outputs();
        CREATE CONSTRAINT TRIGGER metric_values_validate_success AFTER INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_values DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.validate_metric_evaluation_success_outputs();
        CREATE CONSTRAINT TRIGGER metric_artifacts_validate_success AFTER INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_artifacts DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.validate_metric_evaluation_success_outputs();
        DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api') THEN EXECUTE 'REVOKE ALL PRIVILEGES ON TABLE benchmarks_v2.benchmark_observations, benchmarks_v2.preprocessing_artifacts, benchmarks_v2.metric_evaluations, benchmarks_v2.metric_values, benchmarks_v2.metric_artifacts, benchmarks_v2.metric_values_by_bucket FROM api'; END IF; END; $$;
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
        DROP TRIGGER IF EXISTS preprocessing_artifacts_immutable ON benchmarks_v2.preprocessing_artifacts;
        DROP FUNCTION IF EXISTS benchmarks_v2.guard_immutable_preprocessing_artifact();
        DROP TRIGGER IF EXISTS metric_evaluations_validate_transition ON benchmarks_v2.metric_evaluations;
        DROP FUNCTION IF EXISTS benchmarks_v2.validate_metric_transition();
        DROP TABLE IF EXISTS benchmarks_v2.metric_values_by_bucket;
        DROP TABLE IF EXISTS benchmarks_v2.metric_artifacts;
        DROP TABLE IF EXISTS benchmarks_v2.metric_values;
        DROP TABLE IF EXISTS benchmarks_v2.metric_evaluations;
        DROP TABLE IF EXISTS benchmarks_v2.preprocessing_artifacts;
        DROP TABLE IF EXISTS benchmarks_v2.benchmark_observations;
        """
    )
