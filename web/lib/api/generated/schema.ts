/**
 * This file mirrors the FastAPI OpenAPI contract used by the web client.
 * Refresh with `pnpm codegen` when API response schemas change.
 */

export interface paths {
  "/v1/results/aggregates": {
    get: {
      parameters: {
        query: {
          benchmark: components["schemas"]["BenchmarkLiteral"];
          window?: components["schemas"]["WindowLiteral"];
          dataset?: string | null;
          schedule_period?: number;
        };
      };
      responses: {
        200: {
          content: {
            "application/json": components["schemas"]["AggregatesResponse"];
          };
        };
      };
    };
  };
  "/v1/results/aggregates/by-dataset": {
    get: {
      parameters: {
        query: {
          benchmark: components["schemas"]["BenchmarkLiteral"];
          window?: components["schemas"]["WindowLiteral"];
        };
      };
      responses: {
        200: {
          content: {
            "application/json": components["schemas"]["AggregatesByDatasetResponse"];
          };
        };
      };
    };
  };
  "/v1/pricing": {
    get: {
      parameters: {
        query: {
          benchmark: components["schemas"]["BenchmarkLiteral"];
        };
      };
      responses: {
        200: {
          content: {
            "application/json": components["schemas"]["PricingResponse"];
          };
        };
      };
    };
  };
}

export interface components {
  schemas: {
    AggregatesByDatasetResponse: {
      benchmark: components["schemas"]["BenchmarkLiteral"];
      window: components["schemas"]["WindowLiteral"];
      blocks: components["schemas"]["DatasetAggregates"][];
    };
    AggregatesResponse: {
      benchmark: components["schemas"]["BenchmarkLiteral"];
      window: components["schemas"]["WindowLiteral"];
      dataset: string;
      datasets: string[];
      model_stats: components["schemas"]["ModelStatEntry"][];
      series: components["schemas"]["SeriesPoint"][];
    };
    BenchmarkLiteral: "STT" | "TTS" | "S2S";
    DatasetAggregates: {
      dataset: string;
      model_stats: components["schemas"]["ModelStatEntry"][];
    };
    LeaderboardEntry: {
      provider: string;
      model: string;
      avg: number;
      p50: number;
      p95: number;
      n: number;
    };
    LeaderboardResponse: {
      metric: "WER" | "TTFA" | "TTFT" | "TTFS" | "V2V";
      window: components["schemas"]["WindowLiteral"];
      entries: components["schemas"]["LeaderboardEntry"][];
    };
    ModelInfo: {
      model: string;
      disabled?: boolean;
      tags?: components["schemas"]["ModelTagOut"][];
    };
    ModelStatEntry: {
      provider: string;
      model: string;
      metric_type: string;
      avg_value: number;
      stddev_value: number;
      p25: number;
      p50: number;
      p75: number;
      p90: number;
      p95: number;
      p99: number;
      min_value: number;
      max_value: number;
      sample_count: number;
      /** WER only: avg_value split by error type, in percentage points summing
       *  to avg_value. Null on other metrics and on WER groups whose rows
       *  predate the breakdown. */
      wer_insertions_pct?: number | null;
      wer_deletions_pct?: number | null;
      wer_substitutions_pct?: number | null;
    };
    ModelTagOut: {
      category: components["schemas"]["TagCategory"];
      value: string;
      label: string;
    };
    ConversionOut: {
      in_tokens_per_min?: number | null;
      out_tokens_per_min?: number | null;
      chars_per_sec?: number | null;
      sample_count: number;
      window: string;
    };
    NativeRateOut: {
      billing_unit: string;
      rate_usd: number;
      plan_assumption?: string | null;
    };
    PriceHistoryPoint: {
      normalized_usd?: number | null;
      effective_at: string;
      superseded_at?: string | null;
    };
    PricingEntry: {
      provider: string;
      model: string;
      normalized_usd?: number | null;
      basis?: ("list_price" | "list_price_measured_conversion") | null;
      native_rates: components["schemas"]["NativeRateOut"][];
      conversion?: components["schemas"]["ConversionOut"] | null;
      as_of: string;
      source_url: string;
      history: components["schemas"]["PriceHistoryPoint"][];
    };
    PricingResponse: {
      benchmark: components["schemas"]["BenchmarkLiteral"];
      unit_label: "USD per 1,000 minutes" | "USD per 1M characters";
      entries: components["schemas"]["PricingEntry"][];
    };
    ProviderInfo: {
      provider: string;
      models: components["schemas"]["ModelInfo"][];
      modes?: string[] | null;
    };
    ProvidersResponse: {
      stt: components["schemas"]["ProviderInfo"][];
      tts: components["schemas"]["ProviderInfo"][];
      s2s: components["schemas"]["ProviderInfo"][];
      tag_categories: components["schemas"]["TagCategoryOut"][];
    };
    RunOut: {
      id: number;
      started_at: string;
      finished_at: string | null;
      status: "RUNNING" | "SUCCEEDED" | "PARTIAL" | "FAILED";
      runner_sha: string;
      dataset_id: string;
      dataset_sha256: string;
      error: string | null;
    };
    RunsResponse: {
      runs: components["schemas"]["RunOut"][];
      next_before?: number | null;
    };
    SeriesPoint: {
      provider: string;
      model: string;
      metric_type: string;
      scheduled_at: string;
      min_value: number;
      p25: number;
      p50: number;
      p75: number;
      max_value: number;
      value_sum: number;
      sample_count: number;
    };
    TagCategory:
      | "type"
      | "mode"
      | "host"
      | "creator"
      | "features"
      | "source"
      | "licensing"
      | "deployment"
      | "region";
    TagCategoryOut: {
      category: components["schemas"]["TagCategory"];
      label: string;
      provider_valued?: boolean;
    };
    WindowLiteral: "24h" | "7d" | "30d";
  };
}
