// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from "recharts";
import CustomBarTooltip from "@/components/charts/tooltips/BarTooltip";
import CustomBarChartTick from "@/components/charts/CustomBarChartTick";
import SectionHeader from "@/components/shared/SectionHeader";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";

const AccuracyBarSection: React.FC = () => {
  const {
    werDescription: description,
    werBarDataWithColors,
    getProviderForModel,
    isMobile,
    handleWERBarClick,
  } = useDashboard();

  const themeColors = useThemeColors();

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
      <div className="w-full relative z-[2] border border-border-secondary rounded-lg bg-white p-8">
        <SectionHeader
          label="Accuracy by Model"
          description={description}
          stat={{ label: "Avg WER", value: `${avgWER.toFixed(1)}%` }}
        />
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={werBarDataWithColors}
              margin={{
                top: 20,
                right: 30,
                left: 20,
                bottom: isMobile ? 80 : 60,
              }}
            >
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
                height={isMobile ? 100 : 80}
                interval={0}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) => `${value}%`}
                label={{
                  value: "Average WER (%)",
                  angle: -90,
                  position: "insideLeft",
                  style: {
                    textAnchor: "middle",
                    fill: themeColors.axisText,
                    fontSize: "14px",
                  },
                }}
              />
              <Tooltip content={<CustomBarTooltip getProviderForModel={getProviderForModel} />} cursor={false} />
              <Bar
                dataKey="averageWER"
                stroke={themeColors.barStroke}
                strokeWidth={1}
                radius={[4, 4, 0, 0]}
                onClick={handleWERBarClick}
                label={{
                  position: "top",
                  fill: themeColors.label,
                  fontSize: 12,
                  formatter: (value: number) => `${value.toFixed(1)}%`,
                }}
                style={{
                  filter:
                    "drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1))",
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
      </div>
    </div>
  );
};

export default AccuracyBarSection;
