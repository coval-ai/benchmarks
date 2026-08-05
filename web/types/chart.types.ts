// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { BoxPlotData } from './benchmark.types';

export interface BoxPlotProps {
  data: BoxPlotData;
  width?: number;
  height?: number;
  getModelColor: (model: string) => string;
  getProviderForModel: (model: string) => string;
  normalizeModelName: (model: string) => string;
  /** Dedicated-inference endpoints: dashed box border + server icon by the label. */
  dedicatedModels?: Set<string>;
  /** Model key -> inference region, only for models served outside our worker's region. */
  crossRegionModels?: Map<string, string>;
  isMobile?: boolean;
}
