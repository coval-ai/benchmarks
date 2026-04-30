// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { BenchmarkData } from "@/types/benchmark.types";
import { formatTimeWithSeconds } from "@/lib/utils/formatters";
import { to15MinuteBucket } from "@/lib/utils/time";

interface GapTooltipProps {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    name: string;
    color: string;
  }>;
  label?: string | number;
  getProviderForModel: (model: string) => string;
  rawData: BenchmarkData[];
}

const CustomGapTooltip: React.FC<GapTooltipProps> = ({ active, payload, label, getProviderForModel, rawData }) => {
  if (!active || !payload || payload.length === 0) return null;

  // Filter out null/undefined values and sort by gap (smallest gap = best performance)
  const validData = payload
    .filter((item) => item.value != null)
    .sort((a, b) => a.value - b.value);

  if (validData.length === 0) return null;

  const labelTimestamp = Number(label);

  return (
    <div
      style={{
        backgroundColor: "var(--color-surface-tooltip)",
        border: "1px solid var(--color-border-secondary)",
        borderRadius: "8px",
        color: "var(--color-text-on-tooltip)",
        padding: "12px",
        minWidth: "400px",
        maxWidth: "600px"
      }}
    >
      <div style={{ display: "flex", alignItems: "center", marginBottom: "8px" }}>
        <p
          style={{ margin: "0", fontWeight: "bold", fontSize: "12px", marginRight: "12px" }}
        >
          First Token
        </p>
        <p
          style={{ margin: "0", fontWeight: "bold", fontSize: "12px" }}
        >
          {formatTimeWithSeconds(Number(label))}
        </p>
      </div>
      <div style={{ fontSize: "11px" }}>
        {validData.map((item, index) => {
          // Extract model name from dataKey (remove '_gap' suffix)
          const modelName = item.dataKey.replace("_gap", "");
          const provider = getProviderForModel(modelName);
          const gapMs = item.value;

          // Get transcript from rawData for this model/timestamp
          const transcript = rawData
            .filter(
              (d) =>
                d.model === modelName &&
                d.metric_type === "TTFT" &&
                to15MinuteBucket(new Date(d.timestamp).getTime()) === labelTimestamp
            )
            .sort(
              (a, b) =>
                new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
            )[0]?.transcript ?? "";

          return (
            <div
              key={item.dataKey}
              style={{
                margin: "6px 0",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", flex: "0 0 auto" }}>
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
                <div>
                  <div
                    style={{
                      color: index === 0 ? "#10B981" : "var(--color-text-on-tooltip)",
                      fontWeight: index === 0 ? "bold" : "normal"
                    }}
                  >
                    #{index + 1} {modelName}
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

              <div style={{ display: "flex", alignItems: "center", flex: "1 1 auto", justifyContent: "space-between", marginLeft: "12px" }}>
                <span
                  style={{
                    color: "var(--color-text-on-tooltip-secondary)",
                    fontWeight: index === 0 ? "bold" : "normal",
                    flexShrink: 0
                  }}
                >
                  {index === 0 ? "FASTEST" : `+${(gapMs / 1000).toFixed(3)}s`}
                </span>

                <span
                  style={{
                    color: "var(--color-text-on-tooltip-secondary)",
                    fontWeight: index === 0 ? "bold" : "normal",
                    marginLeft: "12px",
                    textAlign: "right",
                    flex: "1 1 auto"
                  }}
                >
                  {transcript}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CustomGapTooltip;
