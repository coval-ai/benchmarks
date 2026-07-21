// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import Card from "@/components/shared/Card";
import { useDashboard } from "@/contexts/DashboardContext";
import { SamplesCard } from "./SamplesCard";

// S2S headline row: the single latency KeyMetric tile beside the samples card.
// The tile mirrors KeyMetrics.tsx (as DashboardSkeleton already does) so the
// shared component stays untouched for STT/TTS; both cells stretch to level.
export function S2STopRow() {
  const { primaryKeyMetric: metric } = useDashboard();

  return (
    <div className="mb-[0.8rem] grid grid-cols-1 items-stretch gap-[0.8rem] lg:grid-cols-2">
      <Card className="text-left min-w-0 h-full" padding="p-5 lg:p-8">
        <div className="text-[0.9rem] font-light text-text-secondary mb-2">
          {metric.label}
        </div>
        <div className="font-mono text-3xl sm:text-4xl lg:text-5xl font-bold mb-4 break-words leading-tight">
          {metric.displayValue}
        </div>
        {metric.subtitle && (
          <div className="text-text-secondary flex flex-col sm:flex-row sm:items-baseline gap-0.5 sm:gap-2">
            {metric.subtitle.name && (
              <span className="font-medium">{metric.subtitle.name}</span>
            )}
            {metric.subtitle.detail && (
              <span className="text-sm text-text-tertiary">{metric.subtitle.detail}</span>
            )}
          </div>
        )}
      </Card>
      <SamplesCard />
    </div>
  );
}
