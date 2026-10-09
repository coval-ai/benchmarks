# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Explicit, transactional retirement of normalized metric-code storage.

The application must use catalog identities before this migration is selected.
Published summaries are rebuilt at their existing boundary and must compare
exactly to their cached payloads. A stale cache aborts the entire migration.
"""

from __future__ import annotations

from alembic import context, op
from sqlalchemy import text
from sqlalchemy.engine import Connection

revision = "20261007_0044"
down_revision = "20261005_0043"
branch_labels = None
depends_on = None

_TABLES = (
    "metric_evaluations",
    "dashboard_metric_values",
    "metric_values_by_bucket",
    "dashboard_hourly_aggregates",
)
_WINDOWS = (("24h", "24 hours"), ("7d", "7 days"), ("30d", "30 days"))
_ID_KEYS = {
    "metric_evaluations": "observation_id, metric_id, metric_version, evaluation_variant",
    "metric_values_by_bucket": "provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant, value_key, bucket_at",
    "dashboard_hourly_aggregates": "provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant, hour_at",
}

# Frozen definitions from 0018, 0035, 0036 and 0043. Do not import application SQL:
# both upgrade and downgrade must retain their meaning after future releases.
_LIFECYCLE_SQL = """
CREATE OR REPLACE FUNCTION benchmarks_v2.validate_metric_transition() RETURNS trigger AS $$
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
"""

_PROJECTION_SQL = """
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

_SYNC_SQL = """
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
"""

_HOURLY_SYNC_SQL = """
CREATE FUNCTION benchmarks_v2.sync_dashboard_metric_identity() RETURNS trigger
    LANGUAGE plpgsql AS $$
    BEGIN
      -- An UPDATE from either application changes only its own identifier.
      -- Discard the unchanged counterpart before resolving the new identity.
      IF TG_OP = 'UPDATE' THEN
        IF NEW.metric_type IS DISTINCT FROM OLD.metric_type
           AND NEW.metric_id IS NOT DISTINCT FROM OLD.metric_id THEN
          NEW.metric_id := NULL;
        ELSIF NEW.metric_id IS DISTINCT FROM OLD.metric_id
           AND NEW.metric_type IS NOT DISTINCT FROM OLD.metric_type THEN
          NEW.metric_type := NULL;
        END IF;
      END IF;
      IF NEW.metric_id IS NULL AND NEW.metric_type IS NULL THEN
        RAISE EXCEPTION 'a metric code or id is required' USING ERRCODE='23502';
      ELSIF NEW.metric_id IS NULL THEN
        NEW.metric_id := benchmarks_v2.metric_id_for_code(NEW.metric_type);
      ELSIF NEW.metric_type IS NULL THEN
        NEW.metric_type := benchmarks_v2.metric_code_for_id(NEW.metric_id);
      ELSIF NEW.metric_id <> benchmarks_v2.metric_id_for_code(NEW.metric_type) THEN
        RAISE EXCEPTION 'metric code and id must refer to the same definition'
          USING ERRCODE='23514';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER dashboard_hourly_sync_metric_identity
      BEFORE INSERT OR UPDATE OF metric_type, metric_id
      ON benchmarks_v2.dashboard_hourly_aggregates FOR EACH ROW
      EXECUTE FUNCTION benchmarks_v2.sync_dashboard_metric_identity();
"""

_RETRY_SYNC_SQL = """
CREATE OR REPLACE FUNCTION benchmarks_v2.sync_metric_evaluation_identity()
        RETURNS trigger LANGUAGE plpgsql AS $fn$
        BEGIN
          IF NEW.metric_id IS NULL THEN
            NEW.metric_id := benchmarks_v2.metric_id_for_code(NEW.metric_type);
          ELSIF NEW.metric_type IS NULL THEN
            NEW.metric_type := benchmarks_v2.metric_code_for_id(NEW.metric_id);
          ELSIF NEW.metric_id <> benchmarks_v2.metric_id_for_code(NEW.metric_type) THEN
            RAISE EXCEPTION 'metric code and id must refer to the same definition'
              USING ERRCODE='23514';
          END IF;
          IF TG_OP = 'INSERT' THEN
            PERFORM pg_advisory_xact_lock(hashtextextended(
              'metric_evaluations:' ||
              ROW(NEW.observation_id, NEW.metric_id, NEW.metric_version,
                  NEW.evaluation_variant)::text, 0));
          END IF;
          RETURN NEW;
        END $fn$
"""

_PARENT_SQL = """
CREATE OR REPLACE FUNCTION benchmarks_v2.guard_dashboard_metric_parent() RETURNS trigger
           LANGUAGE plpgsql AS $fn$
           DECLARE parent_metric_id BIGINT;
           BEGIN
             SELECT metric_id INTO parent_metric_id
               FROM benchmarks_v2.metric_evaluations WHERE id = NEW.evaluation_id;
             IF FOUND AND parent_metric_id IS DISTINCT FROM NEW.metric_id THEN
               RAISE EXCEPTION 'dashboard projection metric identity disagrees with parent' USING ERRCODE='23514';
             END IF;
             RETURN NEW;
           END $fn$
"""

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
       __METRIC_COLUMN__,metric_version,evaluation_variant,
       mean_value,COALESCE(pooled_value,mean_for_ratio) avg_value,stddev_value,p25,p50,p75,p90,p95,p99,
       min_value,max_value,sample_count,primary_sample_count,
       COALESCE(pooled_insertions_pct,wer_insertions_pct) AS wer_insertions_pct,
       COALESCE(pooled_deletions_pct,wer_deletions_pct) AS wer_deletions_pct,
       COALESCE(pooled_substitutions_pct,wer_substitutions_pct) AS wer_substitutions_pct,
       pooled_value,pooled_insertions_pct,pooled_deletions_pct,
       pooled_substitutions_pct,'{"schema_version" : 1}'::jsonb metadata FROM grouped
"""


def _frozen_sql(bind: Connection, sql: str) -> None:
    bind.execute(text(sql))


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _require_opt_in() -> None:
    cfg = op.get_context().config
    if cfg is not None and cfg.attributes.get("allow_metric_code_cleanup") is True:
        return
    if context.get_x_argument(as_dictionary=True).get("allow_metric_code_cleanup") == "true":
        return
    raise RuntimeError(
        "metric code cleanup is opt-in; pass -x allow_metric_code_cleanup=true after the deployed-consumer and rollback-period gates pass"
    )


def _lock(bind: Connection) -> None:
    bind.exec_driver_sql("SET LOCAL lock_timeout = '10s'")
    bind.exec_driver_sql("SET LOCAL statement_timeout = '10min'")
    # Coordinate with publication, then prevent source changes during snapshots,
    # parity checks, interface replacement and downgrade hydration.
    bind.exec_driver_sql("SELECT pg_advisory_xact_lock(hashtextextended('dashboard_summary', 0))")
    bind.exec_driver_sql("""LOCK TABLE benchmarks_v2.runs, benchmarks_v2.metrics,
        benchmarks_v2.benchmark_observations, benchmarks_v2.metric_evaluations,
        benchmarks_v2.metric_values, benchmarks_v2.dashboard_metric_values,
        benchmarks_v2.metric_values_by_bucket, benchmarks_v2.dashboard_hourly_aggregates,
        benchmarks_v2.dashboard_summary_state IN ACCESS EXCLUSIVE MODE""")


def _check_index(bind: Connection, table: str, columns: str) -> None:
    name = table + "_metric_identity_key"
    row = bind.exec_driver_sql(
        """SELECT i.indisunique, i.indisvalid AND i.indisready,
                  i.indpred IS NULL AND i.indexprs IS NULL AND NOT i.indisreplident,
                  array_agg(a.attname ORDER BY x.ordinality), i.indnkeyatts=i.indnatts
           FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
           JOIN pg_index i ON i.indexrelid=c.oid
           LEFT JOIN LATERAL unnest(i.indkey) WITH ORDINALITY x(attnum,ordinality) ON true
           LEFT JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=x.attnum
           WHERE n.nspname='benchmarks_v2' AND c.relname=%s AND i.indrelid=%s::regclass
           GROUP BY i.indisunique,i.indisvalid,i.indisready,i.indpred,i.indexprs,i.indisreplident,i.indnkeyatts,i.indnatts""",
        (name, f"benchmarks_v2.{table}"),
    ).fetchone()
    if row is None or not all((row[0], row[1], row[2], row[4])) or row[3] != columns.split(", "):
        raise RuntimeError(
            f"required exact valid ID uniqueness index is missing or invalid: {name}"
        )


def _preflight(bind: Connection) -> None:
    for table in _TABLES:
        not_null = bind.exec_driver_sql(
            "SELECT attnotnull FROM pg_attribute WHERE attrelid=%s::regclass AND attname='metric_id' AND NOT attisdropped",
            (f"benchmarks_v2.{table}",),
        ).scalar()
        if not not_null:
            raise RuntimeError(f"{table}.metric_id must be NOT NULL")
        fk = bind.exec_driver_sql(
            """SELECT c.convalidated AND NOT c.condeferrable
               AND c.confrelid='benchmarks_v2.metrics'::regclass AND c.confdeltype='r'
               AND c.conkey=ARRAY[(SELECT attnum FROM pg_attribute WHERE attrelid=c.conrelid AND attname='metric_id')]
               AND c.confkey=ARRAY[(SELECT attnum FROM pg_attribute WHERE attrelid=c.confrelid AND attname='id')]
               FROM pg_constraint c WHERE c.conrelid=%s::regclass AND c.conname=%s AND c.contype='f'""",
            (f"benchmarks_v2.{table}", f"{table}_metric_id_fkey"),
        ).scalar()
        if not fk:
            raise RuntimeError(
                f"required validated catalog foreign key is missing or invalid: {table}"
            )
        bad = bind.exec_driver_sql(
            f"SELECT count(*) FROM benchmarks_v2.{table} t LEFT JOIN benchmarks_v2.metrics m ON m.id=t.metric_id WHERE m.id IS NULL OR m.code IS DISTINCT FROM t.metric_type"
        ).scalar()
        if bad:
            raise RuntimeError(f"{table} contains {bad} mismatched metric identities")
    bad = bind.exec_driver_sql(
        "SELECT count(*) FROM benchmarks_v2.dashboard_metric_values p JOIN benchmarks_v2.metric_evaluations e ON e.id=p.evaluation_id WHERE p.metric_id IS DISTINCT FROM e.metric_id"
    ).scalar()
    if bad:
        raise RuntimeError(f"dashboard projection contains {bad} parent identity mismatches")
    for table, columns in _ID_KEYS.items():
        _check_index(bind, table, columns)


def _view_definition(*, codes: bool) -> str:
    if codes:
        return _VIEW_SQL.replace(
            "__METRIC_COLUMN__",
            "metric_type, benchmarks_v2.metric_id_for_code(metric_type) AS metric_id",
        )
    # The frozen aggregate formulas and grouping dimensions are identical. Only
    # the stored identity and the explicit public TTFA component identities move.
    sql = _VIEW_SQL.replace("e.metric_type", "e.metric_id").replace("p.metric_type", "p.metric_id")
    sql = sql.replace("p(metric_type,value)", "p(metric_id,value)")
    sql = sql.replace("public_metric metric_type", "public_metric metric_id")
    for code in ("TTFA", "TTFARoundtrip", "TTFALeadingSilence", "WER"):
        sql = sql.replace(f"'{code}'", f"benchmarks_v2.metric_id_for_code('{code}')")
    return sql.replace("__METRIC_COLUMN__", "metric_id")


def _replace_views(bind: Connection, *, codes: bool) -> None:
    for name, window in _WINDOWS:
        view = f"benchmarks_v2.normalized_results_{name}"
        owner, populated = bind.exec_driver_sql(
            "SELECT pg_get_userbyid(relowner), relispopulated FROM pg_class WHERE oid=%s::regclass",
            (view,),
        ).one()
        grants = bind.exec_driver_sql(
            """SELECT CASE WHEN a.grantee=0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END,
                      a.privilege_type, a.is_grantable
               FROM pg_class c CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))) a
               WHERE c.oid=%s::regclass""",
            (view,),
        ).all()
        if populated:
            bind.exec_driver_sql(
                f"CREATE TEMP TABLE metric_code_snapshot_{name} ON COMMIT DROP AS SELECT to_jsonb(t)-'metric_type' AS payload FROM {view} t"
            )
        bind.exec_driver_sql(f"DROP MATERIALIZED VIEW {view}")
        bind.exec_driver_sql(
            f"CREATE MATERIALIZED VIEW {view} AS "
            + _view_definition(codes=codes).replace("__WINDOW__", window)
            + " WITH NO DATA"
        )
        identity = "metric_type" if codes else "metric_id"
        bind.exec_driver_sql(
            f"CREATE UNIQUE INDEX normalized_results_{name}_key ON {view} (provider,model,benchmark,dataset_id,{identity},metric_version,evaluation_variant)"
        )
        bind.exec_driver_sql(
            f"CREATE INDEX normalized_results_{name}_lookup ON {view} (benchmark,dataset_id,{identity},metric_version,evaluation_variant)"
        )
        if codes:
            bind.exec_driver_sql(
                f"CREATE INDEX normalized_results_{name}_metric_id_lookup ON {view} (benchmark,dataset_id,metric_id,metric_version,evaluation_variant)"
            )
        if populated:
            bind.exec_driver_sql(f"REFRESH MATERIALIZED VIEW {view}")
            changed = bind.exec_driver_sql(f"""SELECT EXISTS (
                (SELECT payload FROM metric_code_snapshot_{name}
                 EXCEPT ALL SELECT to_jsonb(t)-'metric_type' FROM {view} t)
                UNION ALL
                (SELECT to_jsonb(t)-'metric_type' FROM {view} t
                 EXCEPT ALL SELECT payload FROM metric_code_snapshot_{name})
            )""").scalar()
            if changed:
                raise RuntimeError(
                    f"published summary parity failed for {view}; refresh summaries at the existing as_of boundary before retrying"
                )
        # Recreate the existing privilege set, including grant options and PUBLIC.
        # Default privileges of the migration role must not widen the interface.
        new_grantees = (
            bind.exec_driver_sql(
                """SELECT DISTINCT a.grantee FROM pg_class c
            CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))) a
            WHERE c.oid=%s::regclass""",
                (view,),
            )
            .scalars()
            .all()
        )
        for grantee in new_grantees:
            role = (
                "PUBLIC"
                if grantee == 0
                else _quote(
                    str(bind.exec_driver_sql("SELECT pg_get_userbyid(%s)", (grantee,)).scalar())
                )
            )
            bind.exec_driver_sql(f"REVOKE ALL ON {view} FROM {role}")
        for grantee_name, privilege, grantable in grants:
            role = "PUBLIC" if grantee_name == "PUBLIC" else _quote(grantee_name)
            suffix = " WITH GRANT OPTION" if grantable else ""
            bind.exec_driver_sql(f"GRANT {privilege} ON {view} TO {role}{suffix}")
        bind.exec_driver_sql(f"ALTER MATERIALIZED VIEW {view} OWNER TO {_quote(owner)}")


def _drop_code_key(bind: Connection, table: str, *, primary: bool) -> None:
    columns = _ID_KEYS[table].replace("metric_id", "metric_type").split(", ")
    rows = bind.exec_driver_sql(
        """SELECT c.conname, array_agg(a.attname ORDER BY x.ordinality)
           FROM pg_constraint c
           CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY x(attnum,ordinality)
           JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=x.attnum
           WHERE c.conrelid=%s::regclass AND c.contype=%s
           GROUP BY c.conname""",
        (f"benchmarks_v2.{table}", "p" if primary else "u"),
    ).all()
    names = [name for name, actual in rows if actual == columns]
    if len(names) != 1:
        raise RuntimeError(f"required exact compatibility key is missing or ambiguous: {table}")
    bind.exec_driver_sql(f"ALTER TABLE benchmarks_v2.{table} DROP CONSTRAINT {_quote(names[0])}")


def upgrade() -> None:
    _require_opt_in()
    bind = op.get_bind()
    _lock(bind)
    _preflight(bind)
    _replace_views(bind, codes=False)
    _frozen_sql(
        bind,
        _LIFECYCLE_SQL.replace(
            "NEW.metric_type IS DISTINCT FROM OLD.metric_type",
            "NEW.metric_id IS DISTINCT FROM OLD.metric_id",
        ),
    )
    _frozen_sql(
        bind,
        _PROJECTION_SQL.replace("metric_id, metric_type,", "metric_id,").replace(
            "e.metric_id, e.metric_type,", "e.metric_id,"
        ),
    )
    for table, trigger in (
        ("metric_evaluations", "metric_evaluations_sync_metric_identity"),
        ("dashboard_metric_values", "dashboard_metric_values_sync_metric_identity"),
        ("metric_values_by_bucket", "metric_values_by_bucket_sync_metric_identity"),
        ("dashboard_hourly_aggregates", "dashboard_hourly_sync_metric_identity"),
    ):
        bind.exec_driver_sql(f"DROP TRIGGER {trigger} ON benchmarks_v2.{table}")
    for function in (
        "sync_metric_evaluation_identity",
        "sync_normalized_metric_identity",
        "sync_dashboard_metric_identity",
    ):
        bind.exec_driver_sql(f"DROP FUNCTION benchmarks_v2.{function}()")
    _drop_code_key(bind, "metric_evaluations", primary=False)
    for table in ("metric_values_by_bucket", "dashboard_hourly_aggregates"):
        _drop_code_key(bind, table, primary=True)
        bind.exec_driver_sql(
            f"ALTER TABLE benchmarks_v2.{table} ADD CONSTRAINT {table}_pkey PRIMARY KEY USING INDEX {table}_metric_identity_key"
        )
    for table in _TABLES:
        bind.exec_driver_sql(f"ALTER TABLE benchmarks_v2.{table} DROP COLUMN metric_type")
    bind.exec_driver_sql("""CREATE FUNCTION benchmarks_v2.lock_metric_evaluation_identity() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          PERFORM pg_advisory_xact_lock(hashtextextended('metric_evaluations:' ||
            ROW(NEW.observation_id,NEW.metric_id,NEW.metric_version,NEW.evaluation_variant)::text,0));
          RETURN NEW;
        END $$;
        CREATE TRIGGER metric_evaluations_id_retry_lock BEFORE INSERT ON benchmarks_v2.metric_evaluations
          FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.lock_metric_evaluation_identity();""")


def downgrade() -> None:
    bind = op.get_bind()
    _lock(bind)
    # All interfaces are restored in this transaction while writers are blocked.
    # The lifecycle guard cannot treat controlled code hydration as a user update.
    bind.exec_driver_sql(
        "ALTER TABLE benchmarks_v2.metric_evaluations DISABLE TRIGGER metric_evaluations_validate_update"
    )
    bind.exec_driver_sql("SET CONSTRAINTS ALL IMMEDIATE")
    for table in _TABLES:
        bind.exec_driver_sql(f"ALTER TABLE benchmarks_v2.{table} ADD COLUMN metric_type TEXT")
        bind.exec_driver_sql(
            f"UPDATE benchmarks_v2.{table} t SET metric_type=m.code FROM benchmarks_v2.metrics m WHERE m.id=t.metric_id"
        )
        bind.exec_driver_sql(
            f"ALTER TABLE benchmarks_v2.{table} ALTER COLUMN metric_type SET NOT NULL"
        )
    for table in ("metric_evaluations", "metric_values_by_bucket"):
        bind.exec_driver_sql(f"ALTER TABLE benchmarks_v2.{table} ADD CHECK (metric_type <> '')")
    _frozen_sql(bind, _LIFECYCLE_SQL)
    _frozen_sql(bind, _PROJECTION_SQL)
    _frozen_sql(bind, _SYNC_SQL)
    _frozen_sql(bind, _RETRY_SYNC_SQL)
    _frozen_sql(bind, _HOURLY_SYNC_SQL)
    _frozen_sql(bind, _PARENT_SQL)
    bind.exec_driver_sql(
        "ALTER TABLE benchmarks_v2.metric_evaluations ENABLE TRIGGER metric_evaluations_validate_update"
    )
    bind.exec_driver_sql(
        "DROP TRIGGER metric_evaluations_id_retry_lock ON benchmarks_v2.metric_evaluations"
    )
    bind.exec_driver_sql("DROP FUNCTION benchmarks_v2.lock_metric_evaluation_identity()")
    bind.exec_driver_sql(
        "ALTER TABLE benchmarks_v2.metric_evaluations ADD UNIQUE (observation_id,metric_type,metric_version,evaluation_variant)"
    )
    for table in ("metric_values_by_bucket", "dashboard_hourly_aggregates"):
        bind.exec_driver_sql(f"ALTER TABLE benchmarks_v2.{table} DROP CONSTRAINT {table}_pkey")
        bind.exec_driver_sql(
            f"ALTER TABLE benchmarks_v2.{table} ADD PRIMARY KEY ({_ID_KEYS[table].replace('metric_id', 'metric_type')})"
        )
        bind.exec_driver_sql(
            f"CREATE UNIQUE INDEX {table}_metric_identity_key ON benchmarks_v2.{table} ({_ID_KEYS[table]})"
        )
    _replace_views(bind, codes=True)
