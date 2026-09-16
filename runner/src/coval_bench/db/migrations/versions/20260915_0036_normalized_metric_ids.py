# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Add catalog identities to normalized evaluation and dashboard storage.

This is the compatibility stage: the columns remain nullable so historical
rows can be hydrated by the dedicated backfill before the enforcement release.
"""

from __future__ import annotations

from alembic import op

revision = "20260915_0036"
down_revision = "20260914_0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.metric_evaluations ADD COLUMN metric_id BIGINT;
        ALTER TABLE benchmarks_v2.dashboard_metric_values ADD COLUMN metric_id BIGINT;
        ALTER TABLE benchmarks_v2.metric_values_by_bucket ADD COLUMN metric_id BIGINT;

        ALTER TABLE benchmarks_v2.metric_evaluations
          ADD CONSTRAINT metric_evaluations_metric_id_fkey
          FOREIGN KEY (metric_id) REFERENCES benchmarks_v2.metrics(id) ON DELETE RESTRICT NOT VALID;
        ALTER TABLE benchmarks_v2.dashboard_metric_values
          ADD CONSTRAINT dashboard_metric_values_metric_id_fkey
          FOREIGN KEY (metric_id) REFERENCES benchmarks_v2.metrics(id) ON DELETE RESTRICT NOT VALID;
        ALTER TABLE benchmarks_v2.metric_values_by_bucket
          ADD CONSTRAINT metric_values_by_bucket_metric_id_fkey
          FOREIGN KEY (metric_id) REFERENCES benchmarks_v2.metrics(id) ON DELETE RESTRICT NOT VALID;

        CREATE OR REPLACE FUNCTION benchmarks_v2.sync_metric_evaluation_identity()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.metric_id IS NULL THEN
            NEW.metric_id := benchmarks_v2.metric_id_for_code(NEW.metric_type);
          ELSIF NEW.metric_type IS NULL THEN
            NEW.metric_type := benchmarks_v2.metric_code_for_id(NEW.metric_id);
          ELSIF NEW.metric_id <> benchmarks_v2.metric_id_for_code(NEW.metric_type) THEN
            RAISE EXCEPTION 'metric code and id must refer to the same definition'
              USING ERRCODE='23514';
          END IF;
          RETURN NEW;
        END $$;
        CREATE TRIGGER metric_evaluations_sync_metric_identity
          BEFORE INSERT OR UPDATE
          ON benchmarks_v2.metric_evaluations FOR EACH ROW
          EXECUTE FUNCTION benchmarks_v2.sync_metric_evaluation_identity();

        DROP TRIGGER metric_evaluations_validate_transition ON benchmarks_v2.metric_evaluations;
        CREATE TRIGGER metric_evaluations_validate_transition
          BEFORE INSERT OR DELETE ON benchmarks_v2.metric_evaluations FOR EACH ROW
          EXECUTE FUNCTION benchmarks_v2.validate_metric_transition();
        CREATE TRIGGER metric_evaluations_validate_update
          BEFORE UPDATE ON benchmarks_v2.metric_evaluations FOR EACH ROW
          WHEN (NOT (
            OLD.metric_id IS NULL AND NEW.metric_id IS NOT NULL
            AND NEW.metric_id = benchmarks_v2.metric_id_for_code(OLD.metric_type)
            AND (to_jsonb(OLD) - 'metric_id') = (to_jsonb(NEW) - 'metric_id')
          ))
          EXECUTE FUNCTION benchmarks_v2.validate_metric_transition();

        CREATE OR REPLACE FUNCTION benchmarks_v2.sync_normalized_metric_identity()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.metric_id IS NULL THEN
            NEW.metric_id := benchmarks_v2.metric_id_for_code(NEW.metric_type);
          ELSIF NEW.metric_type IS NULL THEN
            NEW.metric_type := benchmarks_v2.metric_code_for_id(NEW.metric_id);
          ELSIF NEW.metric_id <> benchmarks_v2.metric_id_for_code(NEW.metric_type) THEN
            RAISE EXCEPTION 'metric code and id must refer to the same definition'
              USING ERRCODE='23514';
          END IF;
          RETURN NEW;
        END $$;
        CREATE TRIGGER dashboard_metric_values_sync_metric_identity
          BEFORE INSERT OR UPDATE
          ON benchmarks_v2.dashboard_metric_values FOR EACH ROW
          EXECUTE FUNCTION benchmarks_v2.sync_normalized_metric_identity();
        CREATE TRIGGER metric_values_by_bucket_sync_metric_identity
          BEFORE INSERT OR UPDATE
          ON benchmarks_v2.metric_values_by_bucket FOR EACH ROW
          EXECUTE FUNCTION benchmarks_v2.sync_normalized_metric_identity();

        CREATE FUNCTION benchmarks_v2.guard_dashboard_metric_parent() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE parent_metric_id BIGINT;
        BEGIN
          SELECT COALESCE(metric_id, benchmarks_v2.metric_id_for_code(metric_type))
            INTO parent_metric_id
            FROM benchmarks_v2.metric_evaluations WHERE id = NEW.evaluation_id;
          IF FOUND AND parent_metric_id IS DISTINCT FROM NEW.metric_id THEN
            RAISE EXCEPTION 'dashboard projection metric identity disagrees with parent'
              USING ERRCODE='23514';
          END IF;
          RETURN NEW;
        END $$;
        CREATE TRIGGER dashboard_metric_values_z_parent_identity
          BEFORE INSERT OR UPDATE ON benchmarks_v2.dashboard_metric_values
          FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.guard_dashboard_metric_parent();

        CREATE OR REPLACE FUNCTION benchmarks_v2.project_dashboard_metric_values() RETURNS trigger AS $$
        DECLARE existing_metric_id BIGINT;
        BEGIN
          SELECT metric_id INTO existing_metric_id
            FROM benchmarks_v2.dashboard_metric_values WHERE evaluation_id = NEW.id;
          IF FOUND AND existing_metric_id IS DISTINCT FROM NEW.metric_id THEN
            RAISE EXCEPTION 'dashboard projection metric identity disagrees with parent'
              USING ERRCODE='23514';
          END IF;
          INSERT INTO benchmarks_v2.dashboard_metric_values
            (evaluation_id, observation_id, metric_id, metric_type, metric_version, evaluation_variant,
             has_primary_role, value, roundtrip, leading_silence, wer_insertions_pct,
             wer_deletions_pct, wer_substitutions_pct, substitution_count, deletion_count,
             insertion_count, reference_words)
          SELECT e.id, e.observation_id, e.metric_id, e.metric_type, e.metric_version, e.evaluation_variant,
                 COALESCE(BOOL_OR(v.value_role = 'primary'), false),
                 MAX(v.value) FILTER (WHERE v.value_key = 'primary'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'roundtrip'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'leading_silence'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'insertions'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'deletions'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'substitutions'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'substitution_count'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'deletion_count'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'insertion_count'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'reference_words')
          FROM benchmarks_v2.metric_evaluations e
          LEFT JOIN benchmarks_v2.metric_values v ON v.metric_evaluation_id = e.id
          WHERE e.id = NEW.id GROUP BY e.id;
          RETURN NEW;
        END $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TRIGGER metric_evaluations_validate_update ON benchmarks_v2.metric_evaluations;
        DROP TRIGGER metric_evaluations_validate_transition ON benchmarks_v2.metric_evaluations;
        CREATE TRIGGER metric_evaluations_validate_transition
          BEFORE INSERT OR UPDATE OR DELETE ON benchmarks_v2.metric_evaluations FOR EACH ROW
          EXECUTE FUNCTION benchmarks_v2.validate_metric_transition();
        CREATE OR REPLACE FUNCTION benchmarks_v2.project_dashboard_metric_values() RETURNS trigger AS $$
        BEGIN
          INSERT INTO benchmarks_v2.dashboard_metric_values
            (evaluation_id, observation_id, metric_type, metric_version, evaluation_variant,
             has_primary_role, value, roundtrip, leading_silence, wer_insertions_pct,
             wer_deletions_pct, wer_substitutions_pct, substitution_count, deletion_count,
             insertion_count, reference_words)
          SELECT e.id, e.observation_id, e.metric_type, e.metric_version, e.evaluation_variant,
                 COALESCE(BOOL_OR(v.value_role = 'primary'), false),
                 MAX(v.value) FILTER (WHERE v.value_key = 'primary'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'roundtrip'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'leading_silence'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'insertions'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'deletions'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'substitutions'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'substitution_count'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'deletion_count'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'insertion_count'),
                 MAX(v.value) FILTER (WHERE v.value_key = 'reference_words')
          FROM benchmarks_v2.metric_evaluations e
          LEFT JOIN benchmarks_v2.metric_values v ON v.metric_evaluation_id = e.id
          WHERE e.id = NEW.id GROUP BY e.id;
          RETURN NEW;
        END $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS metric_values_by_bucket_sync_metric_identity
          ON benchmarks_v2.metric_values_by_bucket;
        DROP TRIGGER IF EXISTS dashboard_metric_values_sync_metric_identity
          ON benchmarks_v2.dashboard_metric_values;
        DROP TRIGGER IF EXISTS dashboard_metric_values_z_parent_identity
          ON benchmarks_v2.dashboard_metric_values;
        DROP TRIGGER IF EXISTS metric_evaluations_sync_metric_identity
          ON benchmarks_v2.metric_evaluations;
        DROP FUNCTION IF EXISTS benchmarks_v2.sync_normalized_metric_identity();
        DROP FUNCTION IF EXISTS benchmarks_v2.guard_dashboard_metric_parent();
        DROP FUNCTION IF EXISTS benchmarks_v2.sync_metric_evaluation_identity();
        ALTER TABLE benchmarks_v2.metric_values_by_bucket
          DROP CONSTRAINT metric_values_by_bucket_metric_id_fkey, DROP COLUMN metric_id;
        ALTER TABLE benchmarks_v2.dashboard_metric_values
          DROP CONSTRAINT dashboard_metric_values_metric_id_fkey, DROP COLUMN metric_id;
        ALTER TABLE benchmarks_v2.metric_evaluations
          DROP CONSTRAINT metric_evaluations_metric_id_fkey, DROP COLUMN metric_id;
        """
    )
