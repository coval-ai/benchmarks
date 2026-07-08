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
import { normalizeModelName } from "@/lib/utils/formatters";
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
    clickedWERBars,
    handleWERBarClick,
    clearWERBars,
    hasActiveFacets,
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

  const selectedBars = useMemo(
    () => werBarDataWithColors.filter((item) => clickedWERBars.has(item.model)),
    [werBarDataWithColors, clickedWERBars]
  );

  const avgWER = useMemo(() => {
    const source = selectedBars.length > 0 ? selectedBars : werBarDataWithColors;
    if (source.length === 0) return 0;
    const sum = source.reduce((acc, item) => acc + (item.averageWER ?? 0), 0);
    return sum / source.length;
  }, [selectedBars, werBarDataWithColors]);

  // Bar-top value labels are ~28px wide, so on narrow charts they collide.
  // When a bar is too thin, keep the label only on selected bars; the axis
  // and tooltip still carry the values for the rest.
  const barLabel = ({
    x = 0,
    y = 0,
    width = 0,
    value,
    index = 0,
  }: {
    x?: number;
    y?: number;
    width?: number;
    value?: number;
    index?: number;
  }) => {
    const entry = werBarDataWithColors[index];
    if (width < 28 && !(entry && clickedWERBars.has(entry.model))) return <g />;
    return (
      <text
        x={x + width / 2}
        y={y - 8}
        textAnchor="middle"
        fill={themeColors.label}
        fillOpacity={entry?.fillOpacity}
        fontSize={12}
      >
        {`${Number(value).toFixed(1)}%`}
      </text>
    );
  };

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8">
        <SectionHeader
          label="Accuracy by Model"
          description={description}
          hint="Click bar to compare models"
          stat={{
            label:
              selectedBars.length > 0
                ? `Avg WER (${selectedBars.length} selected)`
                : hasActiveFacets
                  ? `Avg WER (${werBarDataWithColors.length} filtered)`
                  : "Avg WER (all models)",
            value:
              werBarDataWithColors.length === 0 ? "—" : `${avgWER.toFixed(1)}%`,
          }}
        />
        {selectedBars.length > 0 && (
          <div className="mb-3 flex flex-wrap items-center gap-1.5">
            {selectedBars.map((item) => (
              <span
                key={item.model}
                className="inline-flex items-center gap-1.5 rounded-full border border-border-primary px-2.5 py-1 text-xs text-text-secondary"
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: item.fill }}
                />
                {normalizeModelName(item.model)}
                <span className="tabular-nums font-medium text-text-primary">
                  {item.averageWER.toFixed(1)}%
                </span>
                <button
                  type="button"
                  aria-label={`Deselect ${normalizeModelName(item.model)}`}
                  onClick={() => handleWERBarClickTracked(item)}
                  className="text-text-tertiary transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40"
                >
                  ✕
                </button>
              </span>
            ))}
            <button
              type="button"
              onClick={clearWERBars}
              className="px-1 text-xs text-text-tertiary transition-colors hover:text-text-primary"
            >
              Clear
            </button>
          </div>
        )}
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
                strokeDasharray="2 2"
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
                width={52}
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) => `${value}%`}
                label={{
                  value: "WER % · lower is better",
                  angle: -90,
                  position: "insideLeft",
                  fill: themeColors.axisText,
                  fontSize: 12,
                  style: { textAnchor: "middle" },
                }}
              />
              <Tooltip content={<CustomBarTooltip getProviderForModel={getProviderForModel} />} cursor={false} />
              <Bar
                dataKey="averageWER"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
                onClick={handleWERBarClickTracked}
                label={barLabel}
                style={{
                  cursor: "pointer",
                }}
              >
                {werBarDataWithColors.map((entry) => (
                  <Cell
                    key={`wer-cell-${entry.model}`}
                    fill={entry.fill}
                    fillOpacity={entry.fillOpacity}
                    role="button"
                    tabIndex={0}
                    aria-label={`${normalizeModelName(entry.model)}: ${entry.averageWER.toFixed(1)}% WER${clickedWERBars.has(entry.model) ? ", selected" : ""}`}
                    onKeyDown={(e: React.KeyboardEvent) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        handleWERBarClickTracked(entry);
                      }
                    }}
                    onMouseDown={(e: React.MouseEvent) => e.preventDefault()}
                  />
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
