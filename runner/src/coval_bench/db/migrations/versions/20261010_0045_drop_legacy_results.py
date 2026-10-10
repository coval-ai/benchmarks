# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Drop the legacy results table, its window matviews, and its bucket rollup."""

from __future__ import annotations

from alembic import op

revision = "20261010_0045"
down_revision = "20261008_0044"
branch_labels = None
depends_on = None

_WINDOWS = {"24h": "24:00:00", "7d": "7 days", "30d": "30 days"}

_WER_PART = (
    "CASE WHEN count(r.wer_insertions_pct) = count(*) AND count(r.wer_deletions_pct) = count(*)"
    " AND count(r.wer_substitutions_pct) = count(*) THEN avg(r.{column}) END AS {column}"
)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '10s'")
    for window in _WINDOWS:
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS benchmarks_v2.results_{window}")
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.results_by_bucket")
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.results")


def downgrade() -> None:
    """Recreate the legacy objects empty; dropped rows are not restored."""
    op.execute("""
    CREATE TABLE benchmarks_v2.results (
      id BIGSERIAL PRIMARY KEY,
      run_id BIGINT NOT NULL REFERENCES benchmarks_v2.runs(id) ON DELETE CASCADE,
      provider TEXT NOT NULL,
      model TEXT NOT NULL,
      voice TEXT,
      benchmark TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S', 'LLM')),
      metric_type TEXT NOT NULL,
      metric_value DOUBLE PRECISION,
      metric_units TEXT,
      audio_filename TEXT,
      transcript TEXT,
      status TEXT NOT NULL CHECK (status IN ('success', 'failed')),
      error TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      http_version TEXT,
      submit_to_headers_ms DOUBLE PRECISION,
      wer_insertions_pct DOUBLE PRECISION,
      wer_deletions_pct DOUBLE PRECISION,
      wer_substitutions_pct DOUBLE PRECISION,
      variant_id TEXT NOT NULL DEFAULT 'pinned',
      transport TEXT,
      test_case_id TEXT
    );
    CREATE INDEX results_benchmark_created_at_idx
      ON benchmarks_v2.results (benchmark, created_at DESC);
    CREATE INDEX results_provider_model_idx
      ON benchmarks_v2.results (provider, model, metric_type, created_at DESC);
    CREATE INDEX results_run_id_idx ON benchmarks_v2.results (run_id);

    CREATE TABLE benchmarks_v2.results_by_bucket (
      provider TEXT NOT NULL,
      model TEXT NOT NULL,
      benchmark TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S', 'LLM')),
      dataset_id TEXT NOT NULL,
      metric_type TEXT NOT NULL,
      bucket_at TIMESTAMPTZ NOT NULL,
      min_value DOUBLE PRECISION NOT NULL,
      p25 DOUBLE PRECISION NOT NULL,
      p50 DOUBLE PRECISION NOT NULL,
      p75 DOUBLE PRECISION NOT NULL,
      max_value DOUBLE PRECISION NOT NULL,
      value_sum DOUBLE PRECISION NOT NULL,
      sample_count INTEGER NOT NULL,
      PRIMARY KEY (provider, model, benchmark, dataset_id, metric_type, bucket_at)
    );
    CREATE INDEX results_by_bucket_series_idx
      ON benchmarks_v2.results_by_bucket (benchmark, dataset_id, bucket_at);
    """)
    dataset = "CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1' ELSE rn.dataset_id END"
    wer = ",\n".join(
        _WER_PART.format(column=column)
        for column in ("wer_insertions_pct", "wer_deletions_pct", "wer_substitutions_pct")
    )
    for window, interval in _WINDOWS.items():
        op.execute(f"""
        CREATE MATERIALIZED VIEW benchmarks_v2.results_{window} AS
        SELECT provider, model, benchmark, dataset_id, metric_type, avg_value, stddev_value,
               min_value, pct[1] AS p25, pct[2] AS p50, pct[3] AS p75, pct[4] AS p90,
               pct[5] AS p95, pct[6] AS p99, max_value, sample_count,
               wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct
        FROM (
          SELECT r.provider, r.model, r.benchmark,
                 COALESCE({dataset}, '__all__') AS dataset_id,
                 r.metric_type,
                 avg(r.metric_value) AS avg_value,
                 COALESCE(stddev_samp(r.metric_value), 0) AS stddev_value,
                 min(r.metric_value) AS min_value,
                 percentile_cont(ARRAY[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
                   WITHIN GROUP (ORDER BY r.metric_value) AS pct,
                 max(r.metric_value) AS max_value,
                 count(*)::int AS sample_count,
                 {wer}
          FROM benchmarks_v2.results r
          JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
          WHERE r.status = 'success' AND rn.status IN ('succeeded', 'partial')
            AND r.metric_value IS NOT NULL
            AND r.created_at >= now() - interval '{interval}'
          GROUP BY GROUPING SETS (
            (r.provider, r.model, r.benchmark, r.metric_type, {dataset}),
            (r.provider, r.model, r.benchmark, r.metric_type)
          )
        ) stats
        WITH NO DATA;
        CREATE UNIQUE INDEX results_{window}_group_key
          ON benchmarks_v2.results_{window} (provider, model, benchmark, dataset_id, metric_type);
        """)
