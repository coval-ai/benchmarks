// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from "recharts";
import CustomBarTooltip from "@/components/charts/tooltips/BarTooltip";
import CustomBarChartTick from "@/components/charts/CustomBarChartTick";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

const AccuracyBarSection: React.FC = () => {
  const {
    werDescription: description,
    werBarDataWithColors,
    getProviderForModel,
    isMobile,
    handleWERBarClick,
  } = useDashboard();

  const themeColors = useThemeColors();
  const mode = useActiveTab();
  const trackChartHover = useChartHoverTracking("wer_bar");

  const handleWERBarClickTracked = (
    data: Parameters<typeof handleWERBarClick>[0]
  ) => {
    if (data?.model) {
      capturePostHogEvent(POSTHOG_EVENTS.dashboardWerBarClicked, {
        surface: `${mode}_dashboard`,
        mode,
        model_id: data.model
      });
    }
    handleWERBarClick(data);
  };

  const avgWER = useMemo(() => {
    if (werBarDataWithColors.length === 0) return 0;
    const sum = werBarDataWithColors.reduce(
      (acc, item) => acc + (item.averageWER ?? 0),
      0
    );
    return sum / werBarDataWithColors.length;
  }, [werBarDataWithColors]);

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8">
        <SectionHeader
          label="Accuracy by Model"
          description={description}
          hint="Click bar to compare models"
          stat={{ label: "Avg WER", value: `${avgWER.toFixed(1)}%` }}
        />
        <div className="h-96" onMouseEnter={trackChartHover}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={werBarDataWithColors}
              margin={{
                top: 20,
                right: 8,
                left: 0,
                bottom: 80,
              }}
            >
              <CartesianGrid
                vertical={false}
                strokeDasharray="0"
                stroke={themeColors.grid}
              />
              <XAxis
                dataKey="model"
                axisLine={false}
                tickLine={false}
                tick={
                  <CustomBarChartTick
                    getProviderForModel={getProviderForModel}
                    isMobile={isMobile}
                  />
                }
                height={100}
                interval={0}
              />
              <YAxis
                width={40}
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip content={<CustomBarTooltip getProviderForModel={getProviderForModel} />} cursor={false} />
              <Bar
                dataKey="averageWER"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
                onClick={handleWERBarClickTracked}
                label={{
                  position: "top",
                  fill: themeColors.label,
                  fontSize: 12,
                  formatter: (value: number) => `${value.toFixed(1)}%`,
                }}
                style={{
                  cursor: "pointer",
                }}
              >
                {werBarDataWithColors.map((entry) => (
                  <Cell key={`wer-cell-${entry.model}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
};

export default AccuracyBarSection;
