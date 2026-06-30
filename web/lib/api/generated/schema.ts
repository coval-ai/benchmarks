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
}

export interface components {
  schemas: {
    AggregatesResponse: {
      benchmark: components["schemas"]["BenchmarkLiteral"];
      window: components["schemas"]["WindowLiteral"];
      model_stats: components["schemas"]["ModelStatEntry"][];
      series: components["schemas"]["SeriesPoint"][];
    };
    BenchmarkLiteral: "STT" | "TTS";
    LeaderboardEntry: {
      provider: string;
      model: string;
      avg: number;
      p50: number;
      p95: number;
      n: number;
    };
    LeaderboardResponse: {
      metric: "WER" | "TTFA" | "TTFT" | "TTFS";
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
    };
    ModelTagOut: {
      category: components["schemas"]["TagCategory"];
      value: string;
      label: string;
    };
    ProviderInfo: {
      provider: string;
      models: components["schemas"]["ModelInfo"][];
      modes?: string[] | null;
    };
    ProvidersResponse: {
      stt: components["schemas"]["ProviderInfo"][];
      tts: components["schemas"]["ProviderInfo"][];
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
    TagCategory: "type" | "mode" | "host" | "lab" | "features" | "source" | "tenancy";
    TagCategoryOut: {
      category: components["schemas"]["TagCategory"];
      label: string;
      provider_valued?: boolean;
    };
    WindowLiteral: "24h" | "7d" | "30d";
  };
}
