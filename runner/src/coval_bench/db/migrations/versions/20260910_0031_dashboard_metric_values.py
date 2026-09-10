# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608

"""Cache the normalized evaluation payload used by dashboard reads."""

from __future__ import annotations

from alembic import op

revision = "20260910_0031"
down_revision = "20260910_0030"
branch_labels = None
depends_on = None


_PROJECTION_COLUMNS = """
 evaluation_id UUID PRIMARY KEY REFERENCES benchmarks_v2.metric_evaluations(id) ON DELETE CASCADE,
 observation_id UUID NOT NULL,
 metric_type TEXT NOT NULL, metric_version TEXT NOT NULL, evaluation_variant TEXT NOT NULL,
 has_primary_role BOOLEAN NOT NULL,
 value DOUBLE PRECISION, roundtrip DOUBLE PRECISION, leading_silence DOUBLE PRECISION,
 wer_insertions_pct DOUBLE PRECISION, wer_deletions_pct DOUBLE PRECISION,
 wer_substitutions_pct DOUBLE PRECISION, substitution_count DOUBLE PRECISION,
 deletion_count DOUBLE PRECISION, insertion_count DOUBLE PRECISION,
 reference_words DOUBLE PRECISION
"""


def upgrade() -> None:
    """Create the projection atomically; retry migration if a writer holds a lock."""
    op.execute(  # noqa: S608
        """
        LOCK TABLE benchmarks_v2.metric_evaluations, benchmarks_v2.metric_values,
                   benchmarks_v2.metric_artifacts IN SHARE ROW EXCLUSIVE MODE NOWAIT;
        CREATE OR REPLACE FUNCTION benchmarks_v2.guard_terminal_metric_payload() RETURNS trigger AS $$
        DECLARE parent_id UUID; old_parent_id UUID; parent_status TEXT;
        BEGIN
          IF TG_OP = 'DELETE' THEN parent_id := OLD.metric_evaluation_id;
          ELSE parent_id := NEW.metric_evaluation_id; END IF;
          IF TG_OP = 'UPDATE' AND OLD.metric_evaluation_id IS DISTINCT FROM NEW.metric_evaluation_id THEN
            old_parent_id := OLD.metric_evaluation_id;
          END IF;
          FOR parent_id IN
            SELECT DISTINCT id FROM unnest(ARRAY[parent_id, old_parent_id]) AS ids(id)
            WHERE id IS NOT NULL ORDER BY id
          LOOP
            SELECT status INTO parent_status FROM benchmarks_v2.metric_evaluations
              WHERE id = parent_id FOR UPDATE;
            IF FOUND AND parent_status IN ('succeeded', 'failed')
              THEN RAISE EXCEPTION 'terminal work payloads are immutable'; END IF;
          END LOOP;
          RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END; $$ LANGUAGE plpgsql;
        CREATE TABLE benchmarks_v2.dashboard_metric_values (
        """
        + _PROJECTION_COLUMNS
        + """
        );
        CREATE INDEX dashboard_metric_values_observation_id
          ON benchmarks_v2.dashboard_metric_values (observation_id);
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
        JOIN benchmarks_v2.metric_values v ON v.metric_evaluation_id = e.id
        WHERE e.status = 'succeeded'
        GROUP BY e.id;
        CREATE FUNCTION benchmarks_v2.project_dashboard_metric_values() RETURNS trigger AS $$
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
        END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER metric_evaluations_dashboard_projection
          AFTER UPDATE OF status ON benchmarks_v2.metric_evaluations
          FOR EACH ROW WHEN (NEW.status = 'succeeded')
          EXECUTE FUNCTION benchmarks_v2.project_dashboard_metric_values();
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api') THEN
            REVOKE ALL PRIVILEGES ON TABLE benchmarks_v2.dashboard_metric_values FROM api;
            GRANT SELECT ON TABLE benchmarks_v2.dashboard_metric_values TO api;
          END IF;
        END $$;
        """
    )


def downgrade() -> None:
    """Remove the projection and restore the pre projection payload guard."""
    op.execute(
        """
        DROP TRIGGER IF EXISTS metric_evaluations_dashboard_projection ON benchmarks_v2.metric_evaluations;
        DROP FUNCTION IF EXISTS benchmarks_v2.project_dashboard_metric_values();
        CREATE OR REPLACE FUNCTION benchmarks_v2.guard_terminal_metric_payload() RETURNS trigger AS $$
        DECLARE parent_status TEXT;
        BEGIN
            SELECT status INTO parent_status FROM benchmarks_v2.metric_evaluations WHERE id = CASE WHEN TG_OP = 'DELETE' THEN OLD.metric_evaluation_id ELSE NEW.metric_evaluation_id END;
            IF FOUND AND parent_status IN ('succeeded', 'failed') THEN RAISE EXCEPTION 'terminal work payloads are immutable'; END IF;
            RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END; $$ LANGUAGE plpgsql;
        DROP TABLE IF EXISTS benchmarks_v2.dashboard_metric_values;
        """
    )
