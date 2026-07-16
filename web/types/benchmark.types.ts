// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export interface ModelsByProvider {
  [provider: string]: string[];
}

export interface TimelineDataPoint {
  timestamp: number;
  timestampLabel: string;
  value?: number;
  benchmark?: string;
  [key: string]: string | number | null | undefined;
}

export interface ScatterDataPoint {
  x: number;
  y: number;
  model: string;
  benchmark: string;
  provider: string;
  count: number;
}

export type LatencyPercentile =
  | "p0"
  | "p25"
  | "p50"
  | "p75"
  | "p90"
  | "p95"
  | "p99"
  | "p100";

export interface ModelHeatmapData {
  model: string;
  latency: Record<LatencyPercentile, number>;
  // Absent on latency-only benchmarks (S2S), or on models with no rows for a
  // pinned WER dataset.
  avgWER?: number;
  werStdDev?: number;
  sampleCount: number;
}

export interface BarDataPoint {
  model: string;
  averageWER: number;
  provider: string;
}

export interface BoxPlotDataPoint {
  model: string;
  provider: string;
  /** Whisker ends are clamped to 1.5x IQR beyond the box. */
  quartiles: {
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
  };
  stats: {
    mean: number;
    std: number;
    count: number;
  };
}

// Aggregate stats are the API's ModelStatEntry — re-exported under the
// chart layer's historical name so consumers track the codegen shape.
export type { ModelStatEntry as ModelStats } from "@/lib/api/client";

export interface BoxPlotData {
  data: BoxPlotDataPoint[];
  globalMin: number;
  globalMax: number;
  metricType: string;
}
