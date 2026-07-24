// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import { useActiveTab } from "@/hooks/useActiveTab";

/**
 * The tab this toggle is showing, or undefined where it stays hidden. Charts
 * pass it to SectionHeader as `exportNote` so a downloaded PNG — which has no
 * toggle to read — still says which tab it was taken on.
 */
export const useMetricTab = () => {
  const activeTab = useActiveTab();
  const { sttMetric } = useDashboard();
  return activeTab === "stt" ? sttMetric : undefined;
};

// Shared TTFS / TTFT switch. Reads and writes the single dashboard-wide metric,
// so every chart that renders it stays in sync. Hidden on TTS (single-metric).
const MetricToggle: React.FC = () => {
  const { sttMetric, setSttMetric } = useDashboard();
  const shown = useMetricTab();

  if (!shown) return null;

  return (
    <div className="mb-4 inline-flex gap-0.5 rounded-lg bg-surface-toggle-inactive p-0.5">
      {(["TTFS", "TTFT"] as const).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => setSttMetric(m)}
          className={
            "rounded-md px-4 py-3 text-sm sm:px-3 sm:py-1 sm:text-xs font-medium transition-colors " +
            (sttMetric === m
              ? "bg-surface-primary text-text-primary shadow-sm"
              : "text-text-secondary hover:text-text-primary")
          }
        >
          {m}
        </button>
      ))}
    </div>
  );
};

export default MetricToggle;
