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
  /**
   * dataKeys whose line is fully clipped out of the current zoom crop. A model
   * counts as "in view" if any part of its curve is visible in the crop — not
   * just its value at the hovered timestamp — so the tooltip surfaces the
   * series the viewer can actually see, matching the legend.
   */
  dimmedKeys?: Set<string>;
  /** Hover mode: only the fastest model plus a pin hint, so the chart stays visible. */
  compact?: boolean;
  /** Optional replacement for the compact hover interaction hint. */
  interactionHint?: string | false;
  /** Caps the scroll area (px); keeps the pinned list from covering the chart on mobile. */
  maxHeight?: number;
  /** Non-timeline charts: shown verbatim instead of the timestamp label. */
  labelText?: string;
  /** Non-latency values: overrides the default "123ms" rendering. */
  formatValue?: (value: number) => string;
}

const CustomTimelineTooltip: React.FC<TimelineTooltipProps> = ({ active, payload, label, getProviderForModel, showDate, dimmedKeys, compact, interactionHint, maxHeight, labelText, formatValue }) => {
  if (!active || !payload || payload.length === 0) return null;

  // Filter out null/undefined values and sort by value (fastest to slowest)
  const validData = payload
    .filter((item) => item.value != null && item.value > 0)
    .sort((a, b) => a.value - b.value); // Ascending = fastest first

  if (validData.length === 0) return null;

  // Ranks come from the overall speed order; in-view models then float above
  // the dimmed ones so the visible series always read first.
  const ranked = validData.map((item, index) => ({
    ...item,
    rank: index + 1,
    inView: !dimmedKeys?.has(item.dataKey),
  }));

  // Compact hover shows one row: the fastest model whose line is in the crop,
  // so a zoom past the global leader still surfaces a visible series.
  const rows = compact
    ? [ranked.find((item) => item.inView) ?? ranked[0]].filter(
        (item): item is (typeof ranked)[number] => item != null
      )
    : [...ranked].sort((a, b) => Number(b.inView) - Number(a.inView));

  // Emphasize the first row shown — the fastest in-view model — rather than the
  // global #1, which a Y-zoom can push into the dimmed group at the bottom.
  const leaderKey = rows[0]?.dataKey;

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
        {labelText ??
          (showDate
            ? `${formatDate(Number(label))} ${formatTimeWithSeconds(Number(label))}`
            : formatTimeWithSeconds(Number(label)))}
      </p>
      <div
        className="tooltip-scroll"
        style={
          compact
            ? { fontSize: "11px" }
            : {
                fontSize: "11px",
                maxHeight: `${maxHeight ?? 300}px`,
                overflowY: "scroll",
                paddingRight: "8px",
                touchAction: "pan-y",
                overscrollBehavior: "contain",
                WebkitOverflowScrolling: "touch",
              }
        }
      >
        {rows.map((item) => {
          // Extract model name from dataKey (remove '_value' suffix)
          const modelName = item.dataKey.replace(/_value$/, "");
          const provider = getProviderForModel(modelName);
          const isLeader = item.dataKey === leaderKey;

          return (
            <div
              key={item.dataKey}
              style={{
                margin: "6px 0",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                opacity: item.inView ? 1 : 0.35
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
                      color: isLeader ? "#10B981" : "var(--color-text-on-tooltip)",
                      fontWeight: isLeader ? "bold" : "normal"
                    }}
                  >
                    #{item.rank} {normalizeModelName(modelName)}
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
                  fontWeight: isLeader ? "bold" : "normal",
                  flexShrink: 0
                }}
              >
                {formatValue ? formatValue(item.value) : `${item.value.toFixed(0)}ms`}
              </span>
            </div>
          );
        })}
      </div>
      {compact && validData.length > 1 && interactionHint !== false && (
        <p
          style={{
            margin: "8px 0 0",
            fontSize: "10px",
            color: "var(--color-text-on-tooltip-secondary)"
          }}
        >
          +{validData.length - 1} more · {interactionHint ?? "click to pin · drag to zoom"}
        </p>
      )}
    </div>
  );
};

export default CustomTimelineTooltip;
