"use client";

import React from "react";
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
import ExpandableDescription from "@/components/shared/ExpandableDescription";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";

const AccuracyBarSection: React.FC = () => {
  const {
    werDescription: description,
    werBarDataWithColors,
    getProviderForModel,
    isMobile,
    sidebarCollapsed,
    handleWERBarClick,
  } = useDashboard();

  const themeColors = useThemeColors();

  return (
    <div className="mb-16">
      <div className="mb-6">
        <h2 className="text-2xl font-light mb-2">Accuracy by Model</h2>
        <ExpandableDescription description={description} />
      </div>
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
                  sidebarCollapsed={sidebarCollapsed}
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
            <Tooltip content={<CustomBarTooltip />} cursor={false} />
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
  );
};

export default AccuracyBarSection;
