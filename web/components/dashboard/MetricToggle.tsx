// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import { useActiveTab } from "@/hooks/useActiveTab";

// Shared TTFS / TTFT switch. Reads and writes the single dashboard-wide metric,
// so every chart that renders it stays in sync. Hidden on TTS (single-metric).
const MetricToggle: React.FC = () => {
  const activeTab = useActiveTab();
  const { sttMetric, setSttMetric } = useDashboard();

  if (activeTab !== "stt") return null;

  return (
    <div className="mb-4 inline-flex gap-0.5 rounded-lg bg-gray-100 p-0.5">
      {(["TTFS", "TTFT"] as const).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => setSttMetric(m)}
          className={
            "rounded-md px-3 py-1 text-xs font-medium transition-colors " +
            (sttMetric === m
              ? "bg-white text-text-primary shadow-sm"
              : "text-gray-500 hover:text-text-primary")
          }
        >
          {m}
        </button>
      ))}
    </div>
  );
};

export default MetricToggle;
