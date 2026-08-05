// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { type CSSProperties, type ReactElement, type ReactNode, type RefObject } from "react";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  type BarProps,
  type BarRectangleItem,
} from "recharts";
import CustomBarChartTick, {
  tickLabelReach,
} from "@/components/charts/CustomBarChartTick";
import { useThemeColors } from "@/hooks/useThemeColors";

const X_AXIS_HEIGHT = 100;
const CHART_BOTTOM_MARGIN = 80;

// A row plotted as one bar. Callers pass the numeric value under `valueKey`
// (e.g. "averageWER" or "instructionScore"); the extra keys ride along for the
// tooltip/label without the chassis needing to know them. Stacked rows must
// NOT carry `fill` — recharts prefers a row's own fill over the segment's.
type QualityBarRow = {
  model: string;
  provider: string;
  fill?: string;
  fillOpacity?: number;
} & Record<string, unknown>;

interface QualityMetricBarsProps {
  /** Rows to plot; each carries its numeric value under `valueKey`. */
  data: readonly QualityBarRow[];
  /** Which numeric field is the bar height (e.g. "averageWER"). */
  valueKey: string;
  /**
   * Stacked segments drawn instead of the single `valueKey` bar (whose key
   * then only sizes the frozen y-axis). Order is bottom-up; the last segment
   * gets the rounded top and the caller's `barLabel`. A function fill is
   * resolved per row's model, so segments can follow the model palette.
   */
  stackSegments?: readonly {
    dataKey: string;
    fill: string | ((model: string) => string);
    fillOpacity?: number;
  }[];
  /** Extra SVG defs (e.g. per-model hatch patterns for segment fills). */
  svgDefs?: ReactNode;
  /** Y-axis tick formatter; defaults to whole percent. */
  tickFormatter?: (value: number) => string;
  /** Rotated y-axis caption, e.g. "WER % · lower is better". */
  yAxisLabel: string;
  /** Value tooltip element (caller owns its content/formatting). */
  tooltip: ReactElement;
  isMobile: boolean;
  getProviderForModel: (model: string) => string;
  /** The per-bar <Cell> list — callers own fill/interaction/aria per bar. */
  children?: ReactNode;
  /** Bar-top value labels; caller-provided so WER can add its markers. */
  barLabel?: BarProps["label"];
  onBarClick?: (bar: BarRectangleItem) => void;
  barStyle?: CSSProperties;
  loading?: boolean;
  onHover?: () => void;
  /** Tooltip active override; defaults to disabled on mobile. */
  tooltipActive?: boolean;
  /** Wrapper ref (WER uses it to anchor the dedicated-inference explainer). */
  wrapRef?: RefObject<HTMLDivElement | null>;
  /** Extra overlay rendered inside the plot frame (e.g. dedicated tip). */
  overlay?: ReactNode;
}

// The shared bar-chart chassis: a frozen y-axis beside a horizontally
// scrollable plot. Used by the WER accuracy chart and the S2S instruction
// chart; both differ only in data, labels, and per-bar cells/tooltip, which
// are passed in so this stays metric-agnostic.
const QualityMetricBars: React.FC<QualityMetricBarsProps> = ({
  data,
  valueKey,
  stackSegments,
  svgDefs,
  tickFormatter = (value) => `${value}%`,
  yAxisLabel,
  tooltip,
  isMobile,
  getProviderForModel,
  children,
  barLabel,
  onBarClick,
  barStyle,
  loading,
  onHover,
  tooltipActive,
  wrapRef,
  overlay,
}) => {
  const themeColors = useThemeColors();
  // Single-bar charts pair mobile taps with click-to-compare, so their tooltip
  // is off there; stacked bars have no click action, so tap-to-inspect stays.
  const activeOverride =
    tooltipActive ?? (isMobile && !stackSegments ? false : undefined);
  // Room the diagonal tick labels need left of the first bar, measured on the
  // providers actually on show since their line is never ellipsized. Shared by
  // both charts, so the instruction bars get the same exact padding as WER.
  const labelReach = tickLabelReach(
    isMobile,
    data.map((row) => getProviderForModel(row.model))
  );

  return (
    <div
      ref={wrapRef}
      className={`relative flex h-96 transition-opacity ${loading ? "opacity-40" : ""}`}
      onMouseEnter={onHover}
      data-export-frame
    >
      <div className="w-[52px] shrink-0" data-chart-axis>
        <ResponsiveContainer width="100%" height="100%" debounce={200}>
          <BarChart
            data={data as QualityBarRow[]}
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
              tickFormatter={tickFormatter}
              label={{
                value: yAxisLabel,
                angle: -90,
                position: "insideLeft",
                fill: themeColors.axisText,
                fontSize: 12,
                style: { textAnchor: "middle" },
              }}
            />
            <Bar dataKey={valueKey} fill="transparent" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="min-w-0 flex-1 overflow-x-auto">
        <ResponsiveContainer
          width="100%"
          height="100%"
          minWidth={data.length * (isMobile ? 56 : 48) + labelReach}
          debounce={200}
        >
          <BarChart
            data={data as QualityBarRow[]}
            accessibilityLayer
            margin={{ top: 20, right: 8, left: 0, bottom: CHART_BOTTOM_MARGIN }}
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
              content={tooltip}
              cursor={false}
              active={activeOverride}
              isAnimationActive={false}
              wrapperStyle={{ pointerEvents: "auto" }}
            />
            {svgDefs}
            {stackSegments ? (
              stackSegments.map((segment, idx) => (
                <Bar
                  key={segment.dataKey}
                  dataKey={segment.dataKey}
                  stackId="stack"
                  fill={
                    typeof segment.fill === "string" ? segment.fill : undefined
                  }
                  fillOpacity={segment.fillOpacity}
                  radius={
                    idx === stackSegments.length - 1 ? [4, 4, 0, 0] : undefined
                  }
                  isAnimationActive={false}
                  label={idx === stackSegments.length - 1 ? barLabel : undefined}
                  style={barStyle}
                >
                  {typeof segment.fill === "function" &&
                    data.map((row) => (
                      <Cell
                        key={`${segment.dataKey}-${row.model}`}
                        fill={(segment.fill as (model: string) => string)(
                          row.model
                        )}
                      />
                    ))}
                </Bar>
              ))
            ) : (
              <Bar
                dataKey={valueKey}
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
                onClick={
                  onBarClick
                    ? (bar: BarRectangleItem) => onBarClick(bar)
                    : undefined
                }
                label={barLabel}
                style={barStyle}
              >
                {children}
              </Bar>
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
      {overlay}
    </div>
  );
};

export default QualityMetricBars;
