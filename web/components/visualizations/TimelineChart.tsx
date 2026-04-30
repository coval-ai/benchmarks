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
  Legend,
  Tooltip
} from "recharts";
import { getModelColor } from "@/lib/utils/colors";
import { formatTime, getLocalTimeZoneAbbr } from "@/lib/utils/formatters";
import { metricDescriptions } from "@/lib/config/metrics";
import CustomTimelineTooltip from "@/components/charts/tooltips/TimelineTooltip";
import ExpandableDescription from "@/components/shared/ExpandableDescription";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";

const TimelineChart: React.FC = () => {
  const activeTab = useActiveTab();
  const {
    selectedModels,
    getWindowedTimelineData,
    getCurrentTimeWindow,
    chartRef,
    isDragging,
    handleMouseDown,
    formatChartLabel,
    getProviderForModel,
  } = useDashboard();

  const themeColors = useThemeColors();
  const windowedTimelineData = getWindowedTimelineData();
  const currentTimeWindow = getCurrentTimeWindow();
  const tzAbbr = getLocalTimeZoneAbbr();
  const xAxisLabel = tzAbbr ? `Time (${tzAbbr})` : "Time";

  return (
    <div className="mb-16">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-2xl font-light mb-2">
            {activeTab === "tts"
              ? "Latency Benchmarks"
              : "Performance Consistency"}
          </h2>
          <ExpandableDescription
            description={
              metricDescriptions[activeTab === "tts" ? "ttfa" : "ttft"]
            }
          />
        </div>
      </div>

      <div
        ref={chartRef}
        className="h-96 pb-6 border border-border-secondary rounded-lg bg-surface-secondary"
        onMouseDown={handleMouseDown}
        style={{
          userSelect: "none",
          cursor: isDragging ? "grabbing" : "grab"
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={windowedTimelineData}>
            <XAxis
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={currentTimeWindow}
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
                  fontSize: "14px"
                }
              }}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: themeColors.axisText, fontSize: 12 }}
              domain={[0, "dataMax"]}
              tickFormatter={(value) => `${(value / 1000).toFixed(1)}s`}
              label={{
                value: `${activeTab === "tts" ? "TTFA" : "TTFT"} (ms)`,
                angle: -90,
                position: "insideLeft",
                style: {
                  textAnchor: "middle",
                  fill: themeColors.axisText,
                  fontSize: "14px"
                }
              }}
            />
            <Tooltip content={<CustomTimelineTooltip getProviderForModel={getProviderForModel} />} />
            {selectedModels.length > 1 && (
              <Legend
                wrapperStyle={{ color: themeColors.axisText, paddingTop: "20px" }}
                iconType="line"
              />
            )}
            {selectedModels.map((model) => (
              <Line
                key={model}
                type="monotone"
                dataKey={`${model}_value`}
                stroke={getModelColor(model)}
                strokeWidth={selectedModels.length === 1 ? 3 : 2}
                dot={false}
                activeDot={{
                  r: selectedModels.length === 1 ? 7 : 6,
                  fill: getModelColor(model)
                }}
                connectNulls={false}
                name={formatChartLabel(model, getProviderForModel(model))}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TimelineChart;
