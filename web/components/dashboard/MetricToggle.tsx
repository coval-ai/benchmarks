// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import { useActiveTab } from "@/hooks/useActiveTab";
import MetricInfo from "@/components/shared/MetricInfo";

// Shared TTFS / TTFT switch. Reads and writes the single dashboard-wide metric,
// so every chart that renders it stays in sync. Hidden on TTS (single-metric).
const MetricToggle: React.FC = () => {
  const activeTab = useActiveTab();
  const { sttMetric, setSttMetric } = useDashboard();

  if (activeTab !== "stt") return null;

  return (
    <div className="mb-4 inline-flex gap-0.5 rounded-lg bg-surface-toggle-inactive p-0.5">
      {(["TTFS", "TTFT"] as const).map((m) => (
        <MetricInfo key={m} metric={m} align="left">
          <button
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
        </MetricInfo>
      ))}
    </div>
  );
};

export default MetricToggle;
