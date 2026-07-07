// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { formatDate, formatTimeWithSeconds, normalizeModelName } from "@/lib/utils/formatters";

interface TimelineTooltipProps {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    name: string;
    color: string;
  }>;
  label?: string | number;
  getProviderForModel: (model: string) => string;
  showDate?: boolean;
  /** Visible Y range when zoomed; models outside it (clipped off-chart) dim. */
  highlightRange?: [number, number];
  /** Hover mode: only the fastest model plus a pin hint, so the chart stays visible. */
  compact?: boolean;
}

const CustomTimelineTooltip: React.FC<TimelineTooltipProps> = ({ active, payload, label, getProviderForModel, showDate, highlightRange, compact }) => {
  if (!active || !payload || payload.length === 0) return null;

  // Filter out null/undefined values and sort by value (fastest to slowest)
  const validData = payload
    .filter((item) => item.value != null && item.value > 0)
    .sort((a, b) => a.value - b.value); // Ascending = fastest first

  if (validData.length === 0) return null;

  const inRange = (value: number) =>
    !highlightRange ||
    (value >= highlightRange[0] && value <= highlightRange[1]);

  // Compact hover shows one row: the fastest model that's actually in view,
  // so a Y-zoom past the global leader still surfaces a visible model.
  const rows = compact
    ? [validData.find((item) => inRange(item.value)) ?? validData[0]].filter(
        (item): item is (typeof validData)[number] => item != null
      )
    : validData;

  return (
    <div
      style={{
        backgroundColor: "var(--color-surface-tooltip)",
        border: "1px solid var(--color-border-secondary)",
        borderRadius: "8px",
        color: "var(--color-text-on-tooltip)",
        padding: "12px",
        minWidth: compact ? "180px" : "250px"
      }}
    >
      <p
        style={{ margin: "0 0 8px 0", fontWeight: "bold", fontSize: "12px" }}
      >
        {showDate
          ? `${formatDate(Number(label))} ${formatTimeWithSeconds(Number(label))}`
          : formatTimeWithSeconds(Number(label))}
      </p>
      <div
        className="tooltip-scroll"
        style={
          compact
            ? { fontSize: "11px" }
            : { fontSize: "11px", maxHeight: "300px", overflowY: "scroll", paddingRight: "8px" }
        }
      >
        {rows.map((item, index) => {
          // Extract model name from dataKey (remove '_value' suffix)
          const modelName = item.dataKey.replace(/_value$/, "");
          const provider = getProviderForModel(modelName);
          const inView = inRange(item.value);

          return (
            <div
              key={item.dataKey}
              style={{
                margin: "6px 0",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                opacity: inView ? 1 : 0.35
              }}
            >
              <div style={{ display: "flex", alignItems: "center", flex: 1 }}>
                <div
                  style={{
                    width: "8px",
                    height: "8px",
                    backgroundColor: item.color,
                    borderRadius: "50%",
                    marginRight: "8px",
                    flexShrink: 0
                  }}
                />
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      color: index === 0 ? "#10B981" : "var(--color-text-on-tooltip)",
                      fontWeight: index === 0 ? "bold" : "normal"
                    }}
                  >
                    #{index + 1} {normalizeModelName(modelName)}
                  </div>
                  <div
                    style={{
                      color: "var(--color-text-on-tooltip-secondary)",
                      fontSize: "10px",
                      marginTop: "1px"
                    }}
                  >
                    {provider}
                  </div>
                </div>
              </div>
              <span
                style={{
                  color: "var(--color-text-on-tooltip-secondary)",
                  marginLeft: "12px",
                  fontWeight: index === 0 ? "bold" : "normal",
                  flexShrink: 0
                }}
              >
                {item.value.toFixed(0)}ms
              </span>
            </div>
          );
        })}
      </div>
      {compact && validData.length > 1 && (
        <p
          style={{
            margin: "8px 0 0",
            fontSize: "10px",
            color: "var(--color-text-on-tooltip-secondary)"
          }}
        >
          +{validData.length - 1} more · click to pin · drag to zoom
        </p>
      )}
    </div>
  );
};

export default CustomTimelineTooltip;
