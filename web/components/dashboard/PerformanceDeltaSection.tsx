// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo, useRef } from "react";
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
import SectionHeader from "@/components/shared/SectionHeader";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";

const PerformanceDeltaSection: React.FC = () => {
  const {
    getModelsWithGapData,
    isDragging,
    handleMouseDown,
    getWindowedGapData,
    getCurrentTimeWindow,
    getTimelineTicks,
    formatChartLabel,
    getProviderForModel,
  } = useDashboard();
  const chartRef = useRef<HTMLDivElement>(null);
  const modelsWithData = getModelsWithGapData();

  const themeColors = useThemeColors();
  const tzAbbr = getLocalTimeZoneAbbr();
  const xAxisLabel = tzAbbr ? `Time (${tzAbbr})` : "Time";

  const windowedGapData = getWindowedGapData();
  const avgGap = useMemo(() => {
    let sum = 0;
    let count = 0;
    for (const point of windowedGapData) {
      const record = point as unknown as Record<string, number>;
      for (const model of modelsWithData) {
        const value = record[`${model}_gap`];
        if (typeof value === "number" && !Number.isNaN(value)) {
          sum += value;
          count += 1;
        }
      }
    }
    return count > 0 ? sum / count : 0;
  }, [windowedGapData, modelsWithData]);

  return (
    <div className="mb-4">
      <div className="w-[75vw] mx-auto p-8 border border-border-secondary rounded-lg bg-white">
        <SectionHeader
          label="Performance Delta Analysis"
          description={metricDescriptions.performanceGap}
          stat={{ label: "Avg Delta", value: `+${avgGap.toFixed(0)} ms` }}
        />

        <div
          ref={chartRef}
          className="h-96"
          onMouseDown={handleMouseDown}
          style={{
            userSelect: "none",
            cursor: isDragging ? "grabbing" : "grab",
          }}
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={windowedGapData}>
              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={getCurrentTimeWindow()}
                ticks={getTimelineTicks()}
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
                    : `+${value.toFixed(0)}ms`
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
                  />
                }
              />
              {modelsWithData.length > 1 && (
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

              {modelsWithData.map((model) => (
                <Line
                  key={model}
                  type="monotone"
                  dataKey={`${model}_gap`}
                  stroke={getModelColor(model)}
                  strokeWidth={modelsWithData.length === 1 ? 3 : 2}
                  dot={false}
                  activeDot={{
                    r: modelsWithData.length === 1 ? 7 : 6,
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
    </div>
  );
};

export default PerformanceDeltaSection;
