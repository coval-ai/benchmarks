// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { ModelHeatmapData, ScatterDataPoint, ViolinPlotData } from './benchmark.types';

export interface HeatmapProps {
  data: ModelHeatmapData[];
  width?: number;
  height?: number;
  formatChartLabel: (model: string, provider: string) => string;
  getProviderForModel: (model: string) => string;
  isMobile?: boolean;
}

export interface SortConfig {
  key: keyof ModelHeatmapData | null;
  direction: "asc" | "desc";
}

export interface ViolinPlotProps {
  data: ViolinPlotData;
  width?: number;
  height?: number;
  getModelColor: (model: string) => string;
  getProviderForModel: (model: string) => string;
  normalizeModelName: (model: string) => string;
  isMobile?: boolean;
  sidebarCollapsed?: boolean;
}

export interface TooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: ScatterDataPoint;
    name: string;
    value: number;
  }>;
  label?: string | number;
}

export interface CustomBarTooltipProps {
  active?: boolean;
  payload?: Array<{
    value: number;
  }>;
  label?: string;
}
