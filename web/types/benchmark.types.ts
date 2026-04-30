// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export interface BenchmarkData {
  provider: string;
  model: string;
  voice: string;
  benchmark: string;
  metric_type: string;
  metric_value: number | null;
  metric_units: string | null;
  audio_filename: string;
  timestamp: string;
  status: string;
  transcript: string;
}

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
}

export interface ModelHeatmapData {
  model: string;
  latencyP25: number;
  latencyP50: number;
  latencyP75: number;
  latencyIQR: number;
  avgWER: number;
  werStdDev: number;
  avgRTF: number; // Will be 0 for TTS models
}

export interface ScatterDataResult {
  points: ScatterDataPoint[];
  p99X: number;
  outlierCount: number;
}

export interface BarDataPoint {
  model: string;
  averageWER: number;
  provider: string;
}

export interface ViolinDataPoint {
  model: string;
  provider: string;
  values: number[];
  density: { value: number; density: number }[];
  quartiles: {
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
    outliers: number[];
  };
  stats: {
    mean: number;
    std: number;
    count: number;
  };
}

export interface ModelStats {
  provider: string;
  model: string;
  metric_type: string;
  avg_value: number;
  stddev_value: number;
  p25: number;
  p50: number;
  p75: number;
  min_value: number;
  max_value: number;
  sample_count: number;
}

export interface ViolinPlotData {
  data: ViolinDataPoint[];
  globalMin: number;
  globalMax: number;
  trueGlobalMax?: number;
  outlierCount?: number;
  cappedAt?: number;
  metricType: string;
}
