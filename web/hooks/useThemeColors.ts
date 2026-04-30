// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useTheme } from "next-themes";
import { useMemo } from "react";

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

const lightColors: ThemeColors = {
  grid: "#E5E7EB",
  axisText: "#6B7280",
  label: "#111827",
  median: "#111827",
  boxFill: "rgba(0, 0, 0, 0.08)",
  barStroke: "#D1D5DB",
  tooltipBg: "#FFFFFF",
  tooltipText: "#111827",
  tooltipSecondary: "#6B7280",
  textPrimary: "#111827",
  textSecondary: "#6B7280",
};

const darkColors: ThemeColors = {
  grid: "#374151",
  axisText: "#9CA3AF",
  label: "#F9FAFB",
  median: "#FFFFFF",
  boxFill: "rgba(255, 255, 255, 0.15)",
  barStroke: "#374151",
  tooltipBg: "#1F2937",
  tooltipText: "#F9FAFB",
  tooltipSecondary: "#9CA3AF",
  textPrimary: "#FFFFFF",
  textSecondary: "#9CA3AF",
};

export function useThemeColors(): ThemeColors {
  const { resolvedTheme } = useTheme();

  return useMemo(() => {
    return resolvedTheme === "dark" ? darkColors : lightColors;
  }, [resolvedTheme]);
}
