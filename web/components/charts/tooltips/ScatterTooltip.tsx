import React from "react";
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
        >{`Model: ${point.model}`}</p>
        <p style={{ margin: 0 }}>{`Provider: ${point.provider}`}</p>
        <p style={{ margin: 0 }}>{`${
          activeTab === "tts" ? "TTFA" : "TTFT"
        }: ${point.x.toFixed(0)}ms`}</p>
        <p style={{ margin: 0 }}>{`WER: ${point.y.toFixed(1)}%`}</p>
      </div>
    );
  }
  return null;
};

export default CustomScatterTooltip;
