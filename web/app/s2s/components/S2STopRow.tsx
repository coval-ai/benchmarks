// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import Card from "@/components/shared/Card";
import { DedicatedInfoIcon } from "@/components/shared/DedicatedInferenceInfo";
import type { KeyMetricData } from "@/components/dashboard/KeyMetrics";
import { useDashboard } from "@/contexts/DashboardContext";
import { SamplesCard } from "./SamplesCard";

// A single KeyMetric tile body, mirroring KeyMetrics.tsx so the S2S headline
// tiles read identically to the STT/TTS dashboards.
function MetricTile({ metric }: { metric: KeyMetricData }) {
  return (
    <>
      <div className="text-[0.9rem] font-light text-text-secondary mb-2">
        {metric.label}
      </div>
      <div className="font-mono text-3xl sm:text-4xl lg:text-5xl font-bold mb-4 break-words leading-tight">
        {metric.displayValue}
      </div>
      {metric.subtitle && (
        <div className="text-text-secondary flex flex-col sm:flex-row sm:items-baseline gap-0.5 sm:gap-2">
          {metric.subtitle.name && (
            <span className="inline-flex items-center gap-1.5 font-medium">
              {metric.subtitle.name}
              {metric.subtitle.dedicated && (
                <DedicatedInfoIcon size={14} className="-m-1 p-1" />
              )}
            </span>
          )}
          {metric.subtitle.detail && (
            <span className="text-sm text-text-tertiary">
              {metric.subtitle.detail}
            </span>
          )}
        </div>
      )}
    </>
  );
}

// S2S headline row: the latency tile (and, when present, the instruction
// adherence tile stacked below it) beside the samples card. The left column
// stretches to the samples card's height; each stacked tile is flex-1 so the
// two tiles plus the gap between them exactly fill that height.
export function S2STopRow() {
  const { primaryKeyMetric, secondaryKeyMetric } = useDashboard();

  return (
    <div className="mb-[0.8rem] grid grid-cols-1 items-stretch gap-[0.8rem] lg:grid-cols-2">
      <div className="flex min-w-0 flex-col gap-[0.8rem]">
        <Card className="text-left min-w-0 flex-1" padding="p-5 lg:p-8">
          <MetricTile metric={primaryKeyMetric} />
        </Card>
        {secondaryKeyMetric && (
          <Card className="text-left min-w-0 flex-1" padding="p-5 lg:p-8">
            <MetricTile metric={secondaryKeyMetric} />
          </Card>
        )}
      </div>
      <SamplesCard />
    </div>
  );
}
