// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

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

// Single light theme tuned for the coval.ai cream palette. These mirror the
// chart-* CSS variables in globals.css (charts read them via JS, not CSS).
const colors: ThemeColors = {
  grid: "#dddbd4",
  axisText: "#515151",
  label: "#0a0a0a",
  median: "#0a0a0a",
  boxFill: "rgba(10, 10, 10, 0.08)",
  barStroke: "#c8c6c2",
  tooltipBg: "#fdfcf8",
  tooltipText: "#0a0a0a",
  tooltipSecondary: "#515151",
  textPrimary: "#0a0a0a",
  textSecondary: "#515151",
};

export function useThemeColors(): ThemeColors {
  return colors;
}
