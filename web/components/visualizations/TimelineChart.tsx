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
  Legend,
  Tooltip
} from "recharts";
import { getModelColor } from "@/lib/utils/colors";
import { formatTime, getLocalTimeZoneAbbr } from "@/lib/utils/formatters";
import { metricDescriptions } from "@/lib/config/metrics";
import CustomTimelineTooltip from "@/components/charts/tooltips/TimelineTooltip";
import SectionHeader from "@/components/shared/SectionHeader";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

const TimelineChart: React.FC = () => {
  const activeTab = useActiveTab();
  const {
    getModelsWithTimelineData,
    getWindowedTimelineData,
    getCurrentTimeWindow,
    getTimelineTicks,
    isDragging,
    handleMouseDown,
    formatChartLabel,
    getProviderForModel,
  } = useDashboard();
  const chartRef = useRef<HTMLDivElement>(null);
  const panFiredRef = useRef(false);
  const trackChartHover = useChartHoverTracking("timeline");

  const handleChartMouseDown = (e: Parameters<typeof handleMouseDown>[0]) => {
    if (!panFiredRef.current) {
      panFiredRef.current = true;
      capturePostHogEvent(POSTHOG_EVENTS.dashboardChartPanned, {
        surface: `${activeTab}_dashboard`,
        mode: activeTab,
        chart: "timeline"
      });
    }
    handleMouseDown(e);
  };

  const themeColors = useThemeColors();
  const modelsWithData = getModelsWithTimelineData();
  const windowedTimelineData = getWindowedTimelineData();
  const currentTimeWindow = getCurrentTimeWindow();
  const tzAbbr = getLocalTimeZoneAbbr();
  const xAxisLabel = tzAbbr ? `Time (${tzAbbr})` : "Time";

  const metricLabel = activeTab === "tts" ? "TTFA" : "TTFT";
  const description =
    metricDescriptions[activeTab === "tts" ? "ttfa" : "ttft"];

  const avgValue = useMemo(() => {
    let sum = 0;
    let count = 0;
    for (const point of windowedTimelineData) {
      const record = point as Record<string, number>;
      for (const model of modelsWithData) {
        const value = record[`${model}_value`];
        if (typeof value === "number" && !Number.isNaN(value)) {
          sum += value;
          count += 1;
        }
      }
    }
    return count > 0 ? sum / count : 0;
  }, [windowedTimelineData, modelsWithData]);

  return (
    <div className="mb-4">
      <div className="w-full p-8 relative z-[2] border border-border-secondary rounded-lg bg-white">
        <SectionHeader
          label={
            activeTab === "tts"
              ? "Latency Benchmarks"
              : "Performance Consistency"
          }
          description={description}
          stat={{
            label: `Avg ${metricLabel}`,
            value: `${avgValue.toFixed(0)} ms`,
          }}
        />

        <div
          ref={chartRef}
          className="h-96"
          onMouseDown={handleChartMouseDown}
          onMouseEnter={trackChartHover}
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
              {modelsWithData.length > 1 && (
                <Legend
                  wrapperStyle={{ color: themeColors.axisText, paddingTop: "20px" }}
                  iconType="line"
                />
              )}
              {modelsWithData.map((model) => (
                <Line
                  key={model}
                  type="monotone"
                  dataKey={`${model}_value`}
                  stroke={getModelColor(model)}
                  strokeWidth={modelsWithData.length === 1 ? 3 : 2}
                  dot={false}
                  activeDot={{
                    r: modelsWithData.length === 1 ? 7 : 6,
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
    </div>
  );
};

export default TimelineChart;
