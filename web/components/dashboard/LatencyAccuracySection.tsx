// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { ScatterDataPoint } from "@/types/benchmark.types";
import { getModelColor } from "@/lib/utils/colors";
import CustomScatterTooltip from "@/components/charts/tooltips/ScatterTooltip";
import ExpandableDescription from "@/components/shared/ExpandableDescription";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useActiveTab } from "@/hooks/useActiveTab";

const LatencyAccuracySection: React.FC = () => {
  const {
    latencyLabel,
    selectedModels,
    scatterData,
    scatterP99X,
    scatterOutlierCount,
  } = useDashboard();

  const activeTab = useActiveTab();
  const themeColors = useThemeColors();

  const description = {
    short: `Raw ${latencyLabel} and WER performance across all measurements`,
    detailed:
      "Every voice AI system faces a fundamental trade-off between speed and accuracy. Faster models might sacrifice precision to deliver quick responses, while more accurate models may take additional processing time to ensure correct results. Choose the model that offers the best balance for your specific use case.",
  };

  return (
    <div className="mb-16">
      <div className="mb-6">
        <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
          <div className="flex-1">
            <h2 className="text-2xl font-light mb-2">Latency vs Accuracy</h2>
            <ExpandableDescription description={description} />
          </div>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart data={scatterData}>
            <XAxis
              dataKey="x"
              type="number"
              name={`${latencyLabel} (ms)`}
              domain={[0, scatterP99X || "dataMax"]}
              axisLine={false}
              tickLine={false}
              tick={{ fill: themeColors.axisText, fontSize: 12 }}
              tickFormatter={(value) => `${Number(value).toFixed(0)}ms`}
              label={{
                value: `${latencyLabel} (ms)`,
                position: "insideBottom",
                offset: -20,
                style: {
                  textAnchor: "middle",
                  fill: themeColors.axisText,
                  fontSize: "14px",
                },
              }}
            />
            <YAxis
              dataKey="y"
              name="WER (%)"
              domain={[0, "dataMax"]}
              axisLine={false}
              tickLine={false}
              tick={{ fill: themeColors.axisText, fontSize: 12 }}
              tickFormatter={(value) => `${value}%`}
              label={{
                value: "WER (%)",
                angle: -90,
                position: "insideLeft",
                style: {
                  textAnchor: "middle",
                  fill: themeColors.axisText,
                  fontSize: "14px",
                },
              }}
            />
            <Tooltip content={<CustomScatterTooltip activeTab={activeTab} />} />
            {selectedModels.map((model: string) => (
              <Scatter
                key={model}
                dataKey="y"
                data={scatterData.filter(
                  (item: ScatterDataPoint) =>
                    item.model === model && item.x <= scatterP99X
                )}
                fill={getModelColor(model)}
                name={model}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {scatterOutlierCount > 0 && (
        <div className="text-center mt-2">
          <p className="text-text-tertiary text-xs">
            {scatterOutlierCount} measurements above{" "}
            {(scatterP99X / 1000).toFixed(1)}s not shown
          </p>
        </div>
      )}
    </div>
  );
};

export default LatencyAccuracySection;
