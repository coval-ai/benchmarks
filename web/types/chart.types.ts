// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { ScatterDataPoint, BoxPlotData } from './benchmark.types';

export interface BoxPlotProps {
  data: BoxPlotData;
  width?: number;
  height?: number;
  getModelColor: (model: string) => string;
  getProviderForModel: (model: string) => string;
  normalizeModelName: (model: string) => string;
  isMobile?: boolean;
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
  getProviderForModel?: (model: string) => string;
}
