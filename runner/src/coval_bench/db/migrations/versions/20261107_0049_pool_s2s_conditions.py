# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Pool every condition of each S2S scenario version in the window views.

``benchmarks_v2.pooled_dataset_id`` is the one place that says which datasets pool:
an S2S scenario's clean caller, noise tiers and platform rooms, of one test-set
version, become ``s2s-<scenario>-all-conditions-v<N>``. The S2S headline reads the
headline scenario's pool. The views here and the rollups both call it. Percentiles
come from the raw values of every member, never from averaged per-dataset rows.
Each condition keeps its own rows, and ``__all__`` is unchanged.

The replacement views are built populated under temporary names and swapped in,
as in 0047, so readers never see an empty view.
"""

from __future__ import annotations

from alembic import op

revision = "20261107_0049"
down_revision = "20261104_0048"
branch_labels = None
depends_on = None

_WINDOWS = (("24h", "24 hours"), ("7d", "7 days"), ("30d", "30 days"))
_INDEX_SUFFIXES = ("key", "lookup", "metric_id_lookup")

_SCENARIOS = (
    "bank",
    "happy-customer",
    "happy-smile",
    "instruction-bank",
    "workflow-bank",
    "id-roulette",
)
_CONDITIONS = (
    "low",
    "medium",
    "hard",
    "extra-hard",
    "quiet-room",
    "office-desk",
    "landline-with-air-conditioner",
)
_DATASET_PATTERN = f"^s2s-({'|'.join(_SCENARIOS)})(-({'|'.join(_CONDITIONS)}))?-v([0-9]+)$"

# Groups: 1 scenario, 3 condition (absent for the clean caller), 4 test-set version.
_POOL_FUNCTION = f"""
CREATE FUNCTION benchmarks_v2.pooled_dataset_id(benchmark TEXT, dataset_id TEXT) RETURNS TEXT
LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE WHEN benchmark = 'S2S' AND dataset_id ~ '{_DATASET_PATTERN}'
              THEN regexp_replace(dataset_id, '{_DATASET_PATTERN}', 's2s-\\1-all-conditions-v\\4')
         END
$$
"""

_SOURCE_SQL = """
WITH evaluations AS (
 SELECT o.provider,o.model,o.benchmark,o.dataset_id,e.metric_type,e.metric_version,e.evaluation_variant,
        e.value,e.roundtrip,e.leading_silence,e.has_primary_role,
        e.substitution_count,e.deletion_count,e.insertion_count,e.reference_words
 FROM benchmarks_v2.dashboard_metric_values e
 JOIN benchmarks_v2.benchmark_observations o ON o.id=e.observation_id
 JOIN benchmarks_v2.runs r ON r.id=o.run_id
 WHERE o.status='succeeded' AND r.status IN ('succeeded','partial')
   AND e.metric_version='v1' AND e.evaluation_variant='default'
   AND (e.metric_type<>'WER' OR e.reference_words IS NOT NULL)
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
        __AGGREGATES__
 FROM public_values GROUP BY GROUPING SETS
 ((provider,model,benchmark,dataset_id,public_metric,metric_version,evaluation_variant),
  (provider,model,benchmark,public_metric,metric_version,evaluation_variant))__POOL_UNION__
)
SELECT provider,model,benchmark,dataset_id,
       metric_type, benchmarks_v2.metric_id_for_code(metric_type) AS metric_id,metric_version,evaluation_variant,
       avg_value,stddev_value,p25,p50,p75,p90,p95,p99,
       min_value,max_value,sample_count,primary_sample_count,
       wer_insertions_pct,wer_deletions_pct,wer_substitutions_pct,'{"schema_version" : 1}'::jsonb metadata
FROM grouped WHERE avg_value IS NOT NULL
"""

_AGGREGATES = """public_metric metric_type,metric_version,evaluation_variant,
        CASE WHEN public_metric='WER'
             THEN 100*SUM(substitution_count+deletion_count+insertion_count)/NULLIF(SUM(reference_words),0)
             ELSE AVG(public_value) END::float8 avg_value,
        COALESCE(STDDEV_SAMP(public_value),0)::float8 stddev_value,
        PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY public_value)::float8 p25,
        PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY public_value)::float8 p50,
        PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY public_value)::float8 p75,
        PERCENTILE_CONT(.9) WITHIN GROUP (ORDER BY public_value)::float8 p90,
        PERCENTILE_CONT(.95) WITHIN GROUP (ORDER BY public_value)::float8 p95,
        PERCENTILE_CONT(.99) WITHIN GROUP (ORDER BY public_value)::float8 p99,
        MIN(public_value)::float8 min_value,MAX(public_value)::float8 max_value,
        COUNT(*)::bigint sample_count,COUNT(*) FILTER (WHERE has_primary_role)::bigint primary_sample_count,
        CASE WHEN public_metric='WER' THEN 100*SUM(insertion_count)/NULLIF(SUM(reference_words),0) END::float8 wer_insertions_pct,
        CASE WHEN public_metric='WER' THEN 100*SUM(deletion_count)/NULLIF(SUM(reference_words),0) END::float8 wer_deletions_pct,
        CASE WHEN public_metric='WER' THEN 100*SUM(substitution_count)/NULLIF(SUM(reference_words),0) END::float8 wer_substitutions_pct"""

# The pooled dataset is its own pass over member rows only, so the other groups cost
# what they did in 0047.
_WITH_POOL = {
    "__POOL_UNION__": """
 UNION ALL
 SELECT provider,model,benchmark,benchmarks_v2.pooled_dataset_id(benchmark,dataset_id) dataset_id,
        __AGGREGATES__
 FROM public_values
 WHERE benchmark='S2S' AND benchmarks_v2.pooled_dataset_id(benchmark,dataset_id) IS NOT NULL
 GROUP BY provider,model,benchmark,benchmarks_v2.pooled_dataset_id(benchmark,dataset_id),
          public_metric,metric_version,evaluation_variant""",
}

# The 0047 definition, so a downgrade drops the pooled rows.
_WITHOUT_POOL = {"__POOL_UNION__": ""}


def _view_sql(window: str, parts: dict[str, str]) -> str:
    sql = _SOURCE_SQL.replace("__WINDOW__", window)
    for marker, text in parts.items():
        sql = sql.replace(marker, text)
    return sql.replace("__AGGREGATES__", _AGGREGATES)


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
    op.execute(_POOL_FUNCTION)
    _swap_views(_WITH_POOL)


def downgrade() -> None:
    _swap_views(_WITHOUT_POOL)
    op.execute("DROP FUNCTION benchmarks_v2.pooled_dataset_id(TEXT, TEXT)")
