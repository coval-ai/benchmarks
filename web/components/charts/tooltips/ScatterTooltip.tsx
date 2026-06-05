// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { normalizeModelName, normalizeSTTProviderName, normalizeTTSProviderName } from "@/lib/utils/formatters";
import { TooltipProps } from "@/types/chart.types";

interface ScatterTooltipProps extends TooltipProps {
  activeTab: "tts" | "stt";
}

const CustomScatterTooltip: React.FC<ScatterTooltipProps> = ({ active, payload, activeTab }) => {
  if (active && payload && payload.length > 0) {
    const point = payload[0]?.payload;
    if (!point) return null;
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
        <p style={{ margin: 0 }}>{`Avg ${
          activeTab === "tts" ? "TTFA" : "TTFT"
        }: ${point.x.toFixed(0)}ms`}</p>
        <p style={{ margin: 0 }}>{`Avg WER: ${point.y.toFixed(1)}%`}</p>
        <p style={{ margin: 0 }}>{`Samples: ${point.count}`}</p>
      </div>
    );
  }
  return null;
};

export default CustomScatterTooltip;
