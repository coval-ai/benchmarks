// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useMemo, useRef } from "react";
import { Server } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  Cell,
  type BarRectangleItem,
  type LabelProps,
} from "recharts";
import CustomBarTooltip from "@/components/charts/tooltips/BarTooltip";
import { normalizeModelName } from "@/lib/utils/formatters";
import CustomBarChartTick, {
  tickLabelReach,
} from "@/components/charts/CustomBarChartTick";
import Card from "@/components/shared/Card";
import { useDedicatedInfoTip } from "@/components/shared/DedicatedInferenceInfo";
import SectionHeader from "@/components/shared/SectionHeader";
import WerBarViewToggle from "@/components/dashboard/WerBarViewToggle";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

const X_AXIS_HEIGHT = 100;
const CHART_BOTTOM_MARGIN = 80;

const AccuracyBarSection: React.FC = () => {
  const {
    werDescription: description,
    werBarView,
    availableWerBarViews,
    werBarDataWithColors,
    getProviderForModel,
    dedicatedModels,
    isMobile,
    clickedWERBars,
    handleWERBarClick,
    clearWERBars,
    hasActiveFacets,
    werBarLoading,
  } = useDashboard();

  const themeColors = useThemeColors();
  const mode = useActiveTab();
  const trackChartHover = useChartHoverTracking("wer_bar");
  const chartWrapRef = useRef<HTMLDivElement>(null);
  const {
    iconHandlers: dedicatedIconHandlers,
    overlay: dedicatedOverlay,
    open: dedicatedTipOpen,
  } = useDedicatedInfoTip(chartWrapRef);

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
  const barLabel = useCallback(({
    x = 0,
    y = 0,
    width = 0,
    value,
    index = 0,
  }: LabelProps) => {
    const entry = werBarDataWithColors[index];
    if (Number(width) < 28 && !(entry && clickedWERBars.has(entry.model))) return <g />;
    const cx = Number(x) + Number(width) / 2;
    return (
      <g opacity={entry?.fillOpacity}>
        <text
          x={cx}
          y={Number(y) - 8}
          textAnchor="middle"
          fill={themeColors.label}
          fontSize={12}
        >
          {`${Number(value).toFixed(1)}%`}
        </text>
        {entry && dedicatedModels.has(entry.model) && (
          // The dedicated marker rides the top of the bar, under the value;
          // hover or tap opens the explainer.
          <g
            {...dedicatedIconHandlers}
            role="button"
            tabIndex={0}
            aria-label="About dedicated inference"
            style={{ cursor: "help" }}
          >
            <Server
              x={cx - 6}
              y={Number(y) + 5}
              size={12}
              color={themeColors.label}
              strokeWidth={2.4}
              aria-hidden
            />
            <rect
              x={cx - 12}
              y={Number(y) - 1}
              width={24}
              height={24}
              fill="transparent"
            />
          </g>
        )}
      </g>
    );
  }, [werBarDataWithColors, clickedWERBars, themeColors.label, dedicatedModels, dedicatedIconHandlers]);

  // WER-based: never rendered on S2S (no WER metric).
  if (mode === "s2s") return null;

  // The toggle buttons carry no tooltips; the active view's blurb rides along
  // in the "About this benchmark" tooltip instead (only when a toggle shows).
  const activeWerView =
    availableWerBarViews.length > 1
      ? availableWerBarViews.find((view) => view.key === werBarView)
      : undefined;
  // Room the diagonal tick labels need left of the first bar, measured on the
  // providers actually on show since their line is never ellipsized.
  const labelReach = tickLabelReach(
    isMobile,
    werBarDataWithColors.map(({ model }) => getProviderForModel(model))
  );

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8">
        <SectionHeader
          label="Accuracy by Model"
          description={description}
          note={
            activeWerView
              ? {
                  term: `${activeWerView.label} view`,
                  text: activeWerView.tooltip,
                }
              : undefined
          }
          exportNote={activeWerView?.label}
          hint="Click bar to compare models"
          exportXLabel="Model"
          exportRows={() =>
            werBarDataWithColors.map(({ model, averageWER }) => ({
              model,
              provider: getProviderForModel(model),
              avg_wer_percent: averageWER,
            }))
          }
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
        <WerBarViewToggle />
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
              className="px-1 text-xs text-text-tertiary transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40"
            >
              Clear
            </button>
          </div>
        )}
        <div
          ref={chartWrapRef}
          className={`relative flex h-96 transition-opacity ${werBarLoading ? "opacity-40" : ""}`}
          onMouseEnter={trackChartHover}
          data-export-frame
        >
          <div className="w-[52px] shrink-0" data-chart-axis>
            <ResponsiveContainer width="100%" height="100%" debounce={200}>
              <BarChart
                data={werBarDataWithColors}
                accessibilityLayer={false}
                margin={{
                  top: 20,
                  right: 0,
                  left: 0,
                  bottom: CHART_BOTTOM_MARGIN + X_AXIS_HEIGHT,
                }}
              >
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
                <Bar dataKey="averageWER" fill="transparent" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="min-w-0 flex-1 overflow-x-auto">
            <ResponsiveContainer
              width="100%"
              height="100%"
              minWidth={werBarDataWithColors.length * (isMobile ? 56 : 48) + labelReach}
              debounce={200}
            >
              <BarChart
                data={werBarDataWithColors}
                accessibilityLayer
                margin={{
                  top: 20,
                  right: 8,
                  left: 0,
                  bottom: CHART_BOTTOM_MARGIN,
                }}
              >
                <CartesianGrid
                  xAxisId={0}
                  yAxisId={0}
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
                  height={X_AXIS_HEIGHT}
                  interval={0}
                  padding={{ left: labelReach }}
                />
                <YAxis hide />
                <Tooltip
                  content={
                    <CustomBarTooltip
                      getProviderForModel={getProviderForModel}
                      dedicatedModels={dedicatedModels}
                    />
                  }
                  cursor={false}
                  // The dedicated marker sits inside the plot, so its open
                  // explainer silences the bar tooltip instead of overlapping it.
                  active={isMobile || dedicatedTipOpen ? false : undefined}
                  isAnimationActive={false}
                  // Recharts defaults the wrapper to pointer-events: none,
                  // which would make the dedicated badge's explainer inert.
                  wrapperStyle={{ pointerEvents: "auto" }}
                />
                <Bar
                  dataKey="averageWER"
                  radius={[4, 4, 0, 0]}
                  isAnimationActive={false}
                  onClick={(bar: BarRectangleItem) =>
                    handleWERBarClickTracked(bar.payload)
                  }
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
                      stroke={dedicatedModels.has(entry.model) ? themeColors.label : undefined}
                      strokeWidth={dedicatedModels.has(entry.model) ? 1.5 : undefined}
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
          {dedicatedOverlay}
        </div>
      </Card>
    </div>
  );
};

export default AccuracyBarSection;
