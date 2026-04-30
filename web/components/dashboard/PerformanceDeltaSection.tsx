// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import { getModelColor } from "@/lib/utils/colors";
import { formatTime, getLocalTimeZoneAbbr } from "@/lib/utils/formatters";
import { metricDescriptions } from "@/lib/config/metrics";
import CustomGapTooltip from "@/components/charts/tooltips/GapTooltip";
import ExpandableDescription from "@/components/shared/ExpandableDescription";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";

const PerformanceDeltaSection: React.FC = () => {
  const {
    selectedModels,
    chartRef,
    isDragging,
    handleMouseDown,
    getWindowedGapData,
    getCurrentTimeWindow,
    formatChartLabel,
    getProviderForModel,
    rawData,
  } = useDashboard();

  const themeColors = useThemeColors();
  const tzAbbr = getLocalTimeZoneAbbr();
  const xAxisLabel = tzAbbr ? `Time (${tzAbbr})` : "Time";

  return (
    <div className="mb-16">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-2xl font-light mb-2">
            Performance Delta Analysis
          </h2>
          <ExpandableDescription
            description={metricDescriptions.performanceGap}
          />
        </div>
      </div>

      <div
        ref={chartRef}
        className="h-96 pb-6 border border-border-secondary rounded-lg bg-surface-secondary"
        onMouseDown={handleMouseDown}
        style={{
          userSelect: "none",
          cursor: isDragging ? "grabbing" : "grab",
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={getWindowedGapData()}>
            <XAxis
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={getCurrentTimeWindow()}
              allowDataOverflow={false}
              axisLine={false}
              tickLine={false}
              tick={{ fill: themeColors.axisText, fontSize: 12 }}
              tickFormatter={(value) => formatTime(value)}
              label={{
                value: xAxisLabel,
                position: "insideBottom",
                offset: -5,
                style: {
                  textAnchor: "middle",
                  fill: themeColors.axisText,
                  fontSize: "14px",
                },
              }}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: themeColors.axisText, fontSize: 12 }}
              domain={[0, "dataMax"]}
              tickFormatter={(value) =>
                value === 0
                  ? "FASTEST"
                  : `+${(value / 1000).toFixed(3)}s`
              }
              label={{
                value: "Performance Gap (ms)",
                angle: -90,
                position: "insideLeft",
                style: {
                  textAnchor: "middle",
                  fill: themeColors.axisText,
                  fontSize: "14px",
                },
              }}
            />
            <Tooltip
              content={
                <CustomGapTooltip
                  getProviderForModel={getProviderForModel}
                  rawData={rawData}
                />
              }
            />
            {selectedModels.length > 1 && (
              <Legend
                wrapperStyle={{
                  color: themeColors.axisText,
                  paddingTop: "20px",
                }}
                iconType="line"
              />
            )}

            <ReferenceLine
              y={0}
              stroke="#10B981"
              strokeDasharray="5 5"
              strokeWidth={2}
              label={{
                value: "FASTEST",
                position: "insideBottomRight",
                fill: "#10B981",
              }}
            />

            {selectedModels.map((model) => (
              <Line
                key={model}
                type="monotone"
                dataKey={`${model}_gap`}
                stroke={getModelColor(model)}
                strokeWidth={selectedModels.length === 1 ? 3 : 2}
                dot={false}
                activeDot={{
                  r: selectedModels.length === 1 ? 7 : 6,
                  fill: getModelColor(model),
                }}
                connectNulls={false}
                name={formatChartLabel(
                  model,
                  getProviderForModel(model)
                )}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PerformanceDeltaSection;
