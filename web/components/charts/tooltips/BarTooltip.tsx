// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import type { TooltipContentProps } from "recharts";
import { normalizeModelName } from "@/lib/utils/formatters";

interface CustomBarTooltipProps extends Partial<Pick<
  TooltipContentProps<number, string>,
  "active" | "payload" | "label"
>> {
  getProviderForModel?: (model: string) => string;
}

const CustomBarTooltip: React.FC<CustomBarTooltipProps> = ({
  active,
  payload,
  label,
  getProviderForModel
}) => {
  if (active && payload && payload.length > 0) {
    const item = payload[0];
    if (item?.dataKey !== "averageWER" || typeof item.value !== "number") return null;
    const value = item.value;
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
        <p style={{ margin: 0, color: "var(--color-text-on-tooltip)" }}>{`Average WER: ${Number(
          value
        ).toFixed(1)}%`}</p>
      </div>
    );
  }
  return null;
};

export default CustomBarTooltip;
