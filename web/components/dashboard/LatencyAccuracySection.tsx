// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { ScatterDataPoint } from "@/types/benchmark.types";
import { getModelColor } from "@/lib/utils/colors";
import CustomScatterTooltip from "@/components/charts/tooltips/ScatterTooltip";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const LatencyAccuracySection: React.FC = () => {
  const { latencyLabel, selectedModels, scatterData } = useDashboard();

  const activeTab = useActiveTab();
  const themeColors = useThemeColors();
  const trackChartHover = useChartHoverTracking("scatter");

  const description = {
    short: `Average ${latencyLabel} and WER per model`,
    detailed:
      "Every voice AI system faces a fundamental trade-off between speed and accuracy. Faster models might sacrifice precision to deliver quick responses, while more accurate models may take additional processing time to ensure correct results. Choose the model that offers the best balance for your specific use case.",
  };

  // Overall run-weighted average, matching the mean over all raw measurements
  const avgLatency = useMemo(() => {
    const totalRuns = scatterData.reduce(
      (acc: number, item: ScatterDataPoint) => acc + item.count,
      0
    );
    if (totalRuns === 0) return 0;
    const sum = scatterData.reduce(
      (acc: number, item: ScatterDataPoint) => acc + item.x * item.count,
      0
    );
    return sum / totalRuns;
  }, [scatterData]);

  // Y domain rounded up to the next 2% step, ticks every 2%
  const { yMax, yTicks } = useMemo(() => {
    const maxWER = scatterData.reduce(
      (acc: number, item: ScatterDataPoint) => Math.max(acc, item.y),
      0
    );
    const max = Math.max(2, Math.ceil((maxWER * 1.1) / 2) * 2);
    const ticks = [];
    for (let t = 0; t <= max; t += 2) ticks.push(t);
    return { yMax: max, yTicks: ticks };
  }, [scatterData]);

  // X ticks at a "nice" step computed from the data, domain rounded up to the last tick
  const { xMax, xTicks } = useMemo(() => {
    const maxLatency = scatterData.reduce(
      (acc: number, item: ScatterDataPoint) => Math.max(acc, item.x),
      0
    );
    const raw = (maxLatency * 1.05) / 5;
    const pow = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const step =
      [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? pow;
    const max = Math.ceil((maxLatency * 1.05) / step) * step;
    const ticks = [];
    for (let t = 0; t <= max; t += step) ticks.push(t);
    return { xMax: max, xTicks: ticks };
  }, [scatterData]);

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8">
        <SectionHeader
          label="Latency vs Accuracy"
          description={description}
          stat={{
            label: `Avg ${latencyLabel}`,
            value: `${avgLatency.toFixed(0)} ms`,
          }}
        />

        <div className="h-64" onMouseEnter={trackChartHover}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart
              margin={{ top: 10, right: 8, left: 0, bottom: 20 }}
            >
              <CartesianGrid
                stroke={themeColors.grid}
                strokeDasharray="2 2"
              />
              <XAxis
                dataKey="x"
                type="number"
                name={`${latencyLabel} (ms)`}
                domain={[0, xMax]}
                ticks={xTicks}
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
                type="number"
                name="WER (%)"
                width={40}
                domain={[0, yMax]}
                ticks={yTicks}
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip content={<CustomScatterTooltip activeTab={activeTab} />} />
              {selectedModels.map((model: string) => (
                <Scatter
                  key={model}
                  data={scatterData.filter(
                    (item: ScatterDataPoint) => item.model === model
                  )}
                  fill={getModelColor(model)}
                  name={model}
                  isAnimationActive={false}
                  shape={(props: { cx?: number; cy?: number; fill?: string }) => (
                    <circle
                      cx={props.cx}
                      cy={props.cy}
                      r={6}
                      fill={props.fill}
                    />
                  )}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
};

export default LatencyAccuracySection;
