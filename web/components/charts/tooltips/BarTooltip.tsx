// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import type { TooltipContentProps } from "recharts";
import { DedicatedBadge } from "@/components/shared/DedicatedInferenceInfo";
import { RegionBadge } from "@/components/shared/InferenceRegionInfo";
import { normalizeModelName } from "@/lib/utils/formatters";
import { WER_BREAKDOWN_LABELS } from "@/lib/utils/werBreakdown";
import type { BarDataPoint, TtfaBreakdownBar } from "@/types/benchmark.types";

interface CustomBarTooltipProps extends Partial<Pick<
  TooltipContentProps<number, string>,
  "active" | "payload" | "label"
>> {
  getProviderForModel?: (model: string) => string;
  /** Dedicated-inference endpoints carry the badge in their tooltip. */
  dedicatedModels?: Set<string>;
  /** Model key -> inference region, for models served outside our worker's region. */
  crossRegionModels?: Map<string, string>;
  /** Bar dataKey to read; defaults to WER so existing callers are unchanged. */
  dataKey?: string;
  /** Value caption; defaults to WER wording. */
  valueLabel?: string;
  /** Value formatter; defaults to one-decimal percent. */
  formatValue?: (value: number) => string;
}

const CustomBarTooltip: React.FC<CustomBarTooltipProps> = ({
  active,
  payload,
  label,
  getProviderForModel,
  dedicatedModels,
  crossRegionModels,
  dataKey = "averageWER",
  valueLabel = "Average WER",
  formatValue = (value) => `${value.toFixed(1)}%`
}) => {
  if (active && payload && payload.length > 0) {
    const item = payload[0];
    if (item?.dataKey !== dataKey || typeof item.value !== "number") return null;
    const value = item.value;
    const breakdown = (item.payload as BarDataPoint | undefined)?.breakdown;
    const modelKey = String(label ?? "");
    const provider = getProviderForModel?.(modelKey);
    const modelLabel = provider
      ? `${provider} ${normalizeModelName(modelKey)}`
      : normalizeModelName(modelKey);
    return (
      <div
        style={{
          backgroundColor: "var(--color-surface-tooltip)",
          border: "1px solid var(--color-border-secondary)",
          borderRadius: "8px",
          padding: "8px 12px"
        }}
      >
        <p
          style={{ margin: 0, fontWeight: "bold", color: "var(--color-text-on-tooltip)" }}
        >{`Model: ${modelLabel}`}</p>
        {dedicatedModels?.has(modelKey) && <DedicatedBadge />}
        <RegionBadge region={crossRegionModels?.get(modelKey)} />
        <p style={{ margin: 0, color: "var(--color-text-on-tooltip)" }}>{`${valueLabel}: ${formatValue(
          value
        )}`}</p>
        {breakdown &&
          WER_BREAKDOWN_LABELS.map(([key, text]) => (
            <p
              key={key}
              style={{
                margin: 0,
                fontSize: 12,
                color: "var(--color-text-on-tooltip)",
                opacity: 0.8,
              }}
            >{`${text}: ${breakdown[key].toFixed(1)}%`}</p>
          ))}
      </div>
    );
  }
  return null;
};

// Tooltip for the stacked TTFA breakdown bars: the total leads, then the two
// segments with their share of it, so the composition reads off the tooltip
// the same way it reads off the bar.
export const TtfaBreakdownTooltip: React.FC<
  Partial<Pick<TooltipContentProps<number, string>, "active" | "payload" | "label">> & {
    getProviderForModel?: (model: string) => string;
  }
> = ({ active, payload, label, getProviderForModel }) => {
  const row = payload?.[0]?.payload as TtfaBreakdownBar | undefined;
  if (!active || !row) return null;
  const modelKey = String(label ?? "");
  const provider = getProviderForModel?.(modelKey);
  const modelLabel = provider
    ? `${provider} ${normalizeModelName(modelKey)}`
    : normalizeModelName(modelKey);
  const parts: [string, number][] = [
    ["Network roundtrip", row.roundtrip],
    ["Leading silence", row.silence],
  ];
  return (
    <div
      style={{
        backgroundColor: "var(--color-surface-tooltip)",
        border: "1px solid var(--color-border-secondary)",
        borderRadius: "8px",
        padding: "8px 12px"
      }}
    >
      <p
        style={{ margin: 0, fontWeight: "bold", color: "var(--color-text-on-tooltip)" }}
      >{`Model: ${modelLabel}`}</p>
      <p style={{ margin: 0, color: "var(--color-text-on-tooltip)" }}>
        {`Avg TTFA: ${Math.round(row.ttfa)} ms`}
      </p>
      {parts.map(([text, value]) => (
        <p
          key={text}
          style={{
            margin: 0,
            fontSize: 12,
            color: "var(--color-text-on-tooltip)",
            opacity: 0.8,
          }}
        >{`${text}: ${Math.round(value)} ms${
          row.ttfa > 0 ? ` (${Math.round((value / row.ttfa) * 100)}%)` : ""
        }`}</p>
      ))}
    </div>
  );
};

export default CustomBarTooltip;
