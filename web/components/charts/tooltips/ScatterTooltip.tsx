// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import type { TooltipContentProps } from "recharts";
import type { ScatterDataPoint } from "@/types/benchmark.types";
import { DedicatedBadge } from "@/components/shared/DedicatedInferenceInfo";
import { normalizeModelName, normalizeSTTProviderName, normalizeTTSProviderName } from "@/lib/utils/formatters";

interface ScatterTooltipProps extends Partial<Pick<
  TooltipContentProps<number, string>,
  "active" | "payload" | "label"
>> {
  activeTab: "tts" | "stt";
  metric: string;
  /** Dedicated-inference endpoints carry the server marker in their tooltip. */
  dedicatedModels?: Set<string>;
}

const CustomScatterTooltip: React.FC<ScatterTooltipProps> = ({ active, payload, activeTab, metric, dedicatedModels }) => {
  if (active && payload && payload.length > 0) {
    const item = payload[0];
    const point = item?.payload as ScatterDataPoint | undefined;
    if (item?.dataKey !== "x" || typeof item.value !== "number" || !item.name || !point) return null;
    return (
      <div
        style={{
          backgroundColor: "var(--color-surface-tooltip)",
          border: "1px solid var(--color-border-secondary)",
          borderRadius: "8px",
          color: "var(--color-text-on-tooltip)",
          padding: "8px 12px"
        }}
      >
        <p
          style={{ margin: 0, fontWeight: "bold" }}
        >{`Model: ${normalizeModelName(point.model)}`}</p>
        <p style={{ margin: 0 }}>{`Provider: ${activeTab === "stt" ? normalizeSTTProviderName(point.provider) : normalizeTTSProviderName(point.provider)}`}</p>
        {dedicatedModels?.has(point.model) && <DedicatedBadge />}
        <p style={{ margin: 0 }}>{`Avg ${metric}: ${point.x.toFixed(0)}ms`}</p>
        <p style={{ margin: 0 }}>{`Avg WER: ${point.y.toFixed(1)}%`}</p>
        <p style={{ margin: 0 }}>{`Samples: ${point.count}`}</p>
      </div>
    );
  }
  return null;
};

export default CustomScatterTooltip;
