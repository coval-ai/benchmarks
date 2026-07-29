// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useTheme } from "next-themes";

export interface ThemeColors {
  grid: string;
  axisText: string;
  label: string;
  median: string;
  boxFill: string;
  barStroke: string;
  tooltipBg: string;
  tooltipText: string;
  tooltipSecondary: string;
  textPrimary: string;
  textSecondary: string;
}

// Charts read these via JS (recharts/d3 fills), not CSS, so they must mirror the
// chart-* CSS variables in globals.css for each theme. Exported so the PNG
// export (which can't call the hook) shares the same source of truth.
export const LIGHT_CHART_COLORS: ThemeColors = {
  grid: "#dbdbd3",
  axisText: "#515151",
  label: "#0f0c0a",
  median: "#0f0c0a",
  boxFill: "rgba(15, 12, 10, 0.08)",
  barStroke: "#c8c6c2",
  tooltipBg: "#f9faf8",
  tooltipText: "#0f0c0a",
  tooltipSecondary: "#515151",
  textPrimary: "#0f0c0a",
  textSecondary: "#515151",
};

const DARK_CHART_COLORS: ThemeColors = {
  grid: "#292720",
  axisText: "#dbdbd3",
  label: "#f9faf8",
  median: "#f9faf8",
  boxFill: "rgba(249, 250, 248, 0.08)",
  barStroke: "#3d3b34",
  tooltipBg: "#292720",
  tooltipText: "#f9faf8",
  tooltipSecondary: "#dbdbd3",
  textPrimary: "#f9faf8",
  textSecondary: "#dbdbd3",
};

export function useThemeColors(): ThemeColors {
  const { resolvedTheme } = useTheme();
  return resolvedTheme === "dark" ? DARK_CHART_COLORS : LIGHT_CHART_COLORS;
}
