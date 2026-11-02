# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Make the pooled ratio the only WER computation in the window views.

Every WER row in the widest window has carried alignment counts since
2026-10-09, so the per-clip mean fallback can never fire again. WER rows
without counts leave the views, ``avg_value`` and the ``wer_*_pct`` split are
ratios of summed counts, and the ``mean_value``/``pooled_*`` columns go.

The replacement views are built populated under temporary names and swapped
in, so readers never see an empty view between this migration and the next
refresh.
"""

from __future__ import annotations

from alembic import op

revision = "20261102_0047"
down_revision = "20261028_0046"
branch_labels = None
depends_on = None

_WINDOWS = (("24h", "24 hours"), ("7d", "7 days"), ("30d", "30 days"))
_INDEX_SUFFIXES = ("key", "lookup", "metric_id_lookup")

_SOURCE_SQL = """
WITH evaluations AS (
 SELECT o.provider,o.model,o.benchmark,o.dataset_id,e.metric_type,e.metric_version,e.evaluation_variant,
        e.value,e.roundtrip,e.leading_silence,e.has_primary_role,__WER_COLUMNS__
 FROM benchmarks_v2.dashboard_metric_values e
 JOIN benchmarks_v2.benchmark_observations o ON o.id=e.observation_id
 JOIN benchmarks_v2.runs r ON r.id=o.run_id
 WHERE o.status='succeeded' AND r.status IN ('succeeded','partial')
   AND e.metric_version='v1' AND e.evaluation_variant='default'__WER_FILTER__
   AND o.captured_at >= (SELECT as_of FROM benchmarks_v2.dashboard_window_state WHERE id=true) - INTERVAL '__WINDOW__'
   AND o.captured_at < (SELECT as_of FROM benchmarks_v2.dashboard_window_state WHERE id=true)
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
        __AVERAGES__
        COALESCE(STDDEV_SAMP(public_value),0)::float8 stddev_value,
        PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY public_value)::float8 p25,
        PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY public_value)::float8 p50,
        PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY public_value)::float8 p75,
        PERCENTILE_CONT(.9) WITHIN GROUP (ORDER BY public_value)::float8 p90,
        PERCENTILE_CONT(.95) WITHIN GROUP (ORDER BY public_value)::float8 p95,
        PERCENTILE_CONT(.99) WITHIN GROUP (ORDER BY public_value)::float8 p99,
        MIN(public_value)::float8 min_value,MAX(public_value)::float8 max_value,
        COUNT(*)::bigint sample_count,COUNT(*) FILTER (WHERE has_primary_role)::bigint primary_sample_count,
        __SPLIT__
 FROM public_values GROUP BY GROUPING SETS
 ((provider,model,benchmark,dataset_id,public_metric,metric_version,evaluation_variant),
  (provider,model,benchmark,public_metric,metric_version,evaluation_variant))
)
SELECT provider,model,benchmark,dataset_id,
       metric_type, benchmarks_v2.metric_id_for_code(metric_type) AS metric_id,metric_version,evaluation_variant,
       __PROJECTION__,'{"schema_version" : 1}'::jsonb metadata FROM grouped__ROW_FILTER__
"""

_POOLED = {
    "__WER_COLUMNS__": "e.substitution_count,e.deletion_count,e.insertion_count,e.reference_words",
    "__WER_FILTER__": "\n   AND (e.metric_type<>'WER' OR e.reference_words IS NOT NULL)",
    "__AVERAGES__": """CASE WHEN public_metric='WER'
             THEN 100*SUM(substitution_count+deletion_count+insertion_count)/NULLIF(SUM(reference_words),0)
             ELSE AVG(public_value) END::float8 avg_value,""",
    "__SPLIT__": """CASE WHEN public_metric='WER' THEN 100*SUM(insertion_count)/NULLIF(SUM(reference_words),0) END::float8 wer_insertions_pct,
        CASE WHEN public_metric='WER' THEN 100*SUM(deletion_count)/NULLIF(SUM(reference_words),0) END::float8 wer_deletions_pct,
        CASE WHEN public_metric='WER' THEN 100*SUM(substitution_count)/NULLIF(SUM(reference_words),0) END::float8 wer_substitutions_pct""",
    "__PROJECTION__": """avg_value,stddev_value,p25,p50,p75,p90,p95,p99,
       min_value,max_value,sample_count,primary_sample_count,
       wer_insertions_pct,wer_deletions_pct,wer_substitutions_pct""",
    "__ROW_FILTER__": " WHERE avg_value IS NOT NULL",
}

# The 0035 definition, kept so a downgrade restores the mean fallback.
_COMPLETE_COUNTS = (
    "COUNT(reference_words)=COUNT(*) AND COUNT(substitution_count)=COUNT(*)"
    " AND COUNT(deletion_count)=COUNT(*) AND COUNT(insertion_count)=COUNT(*)"
)
_COMPLETE_SPLIT = (
    "COUNT(wer_insertions_pct)=COUNT(*) AND COUNT(wer_deletions_pct)=COUNT(*)"
    " AND COUNT(wer_substitutions_pct)=COUNT(*)"
)
_FALLBACK = {
    "__WER_COLUMNS__": "e.wer_insertions_pct,e.wer_deletions_pct,e.wer_substitutions_pct,"
    "e.substitution_count,e.deletion_count,e.insertion_count,e.reference_words",
    "__WER_FILTER__": "",
    "__AVERAGES__": "AVG(public_value)::float8 mean_value,AVG(public_value)::float8 mean_for_ratio,",
    "__SPLIT__": f"""CASE WHEN public_metric='WER' AND {_COMPLETE_COUNTS}
             THEN 100*SUM(substitution_count+deletion_count+insertion_count)/NULLIF(SUM(reference_words),0) END pooled_value,
        CASE WHEN public_metric='WER' AND {_COMPLETE_SPLIT} THEN AVG(wer_insertions_pct) END wer_insertions_pct,
        CASE WHEN public_metric='WER' AND {_COMPLETE_SPLIT} THEN AVG(wer_deletions_pct) END wer_deletions_pct,
        CASE WHEN public_metric='WER' AND {_COMPLETE_SPLIT} THEN AVG(wer_substitutions_pct) END wer_substitutions_pct,
        CASE WHEN public_metric='WER' AND {_COMPLETE_COUNTS}
             THEN 100*SUM(insertion_count)/NULLIF(SUM(reference_words),0) END pooled_insertions_pct,
        CASE WHEN public_metric='WER' AND {_COMPLETE_COUNTS}
             THEN 100*SUM(deletion_count)/NULLIF(SUM(reference_words),0) END pooled_deletions_pct,
        CASE WHEN public_metric='WER' AND {_COMPLETE_COUNTS}
             THEN 100*SUM(substitution_count)/NULLIF(SUM(reference_words),0) END pooled_substitutions_pct""",
    "__PROJECTION__": """mean_value,COALESCE(pooled_value,mean_for_ratio) avg_value,stddev_value,p25,p50,p75,p90,p95,p99,
       min_value,max_value,sample_count,primary_sample_count,
       COALESCE(pooled_insertions_pct,wer_insertions_pct) AS wer_insertions_pct,
       COALESCE(pooled_deletions_pct,wer_deletions_pct) AS wer_deletions_pct,
       COALESCE(pooled_substitutions_pct,wer_substitutions_pct) AS wer_substitutions_pct,
       pooled_value,pooled_insertions_pct,pooled_deletions_pct,pooled_substitutions_pct""",
    "__ROW_FILTER__": "",
}


def _view_sql(window: str, parts: dict[str, str]) -> str:
    sql = _SOURCE_SQL.replace("__WINDOW__", window)
    for marker, text in parts.items():
        sql = sql.replace(marker, text)
    return sql


def _swap_views(parts: dict[str, str]) -> None:
    for name, window in _WINDOWS:
        view = f"benchmarks_v2.results_{name}"
        op.execute(f"CREATE MATERIALIZED VIEW {view}_next AS {_view_sql(window, parts)} WITH DATA")
        op.execute(
            f"CREATE UNIQUE INDEX results_{name}_next_key ON {view}_next (provider,model,benchmark,dataset_id,metric_type,metric_version,evaluation_variant)"
        )
        op.execute(
            f"CREATE INDEX results_{name}_next_lookup ON {view}_next (benchmark,dataset_id,metric_type,metric_version,evaluation_variant)"
        )
        op.execute(
            f"CREATE INDEX results_{name}_next_metric_id_lookup ON {view}_next (benchmark,dataset_id,metric_id,metric_version,evaluation_variant)"
        )
    op.execute("SET LOCAL lock_timeout = '10s'")
    for name, _ in _WINDOWS:
        op.execute(f"DROP MATERIALIZED VIEW benchmarks_v2.results_{name}")
        op.execute(
            f"ALTER MATERIALIZED VIEW benchmarks_v2.results_{name}_next RENAME TO results_{name}"
        )
        for suffix in _INDEX_SUFFIXES:
            op.execute(
                f"ALTER INDEX benchmarks_v2.results_{name}_next_{suffix} RENAME TO results_{name}_{suffix}"
            )
    op.execute("""
      DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='api') THEN
        GRANT SELECT ON benchmarks_v2.results_24h, benchmarks_v2.results_7d,
          benchmarks_v2.results_30d TO api;
      END IF; END $$;
    """)


def upgrade() -> None:
    _swap_views(_POOLED)


def downgrade() -> None:
    _swap_views(_FALLBACK)
