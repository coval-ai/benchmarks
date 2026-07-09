// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import dynamic from "next/dynamic";
import { DashboardProvider } from "@/contexts/DashboardContext";
import { SidebarMenuProvider } from "@/contexts/SidebarMenuContext";
import DashboardLayout from "@/components/layout/DashboardLayout";
import KeyMetrics from "@/components/dashboard/KeyMetrics";
import { ChartSkeleton } from "@/components/dashboard/DashboardSkeleton";

// S2S has a single metric (V2V latency, no WER). The WER-based sections
// (AccuracyBarSection, LatencyAccuracySection) are omitted, and the model
// comparison table (HeatmapSection) is deferred to a later pass — its WER
// columns would need per-column S2S gating.
const TimelineChart = dynamic(
  () => import("@/components/visualizations/TimelineChart"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const BoxPlotSection = dynamic(
  () => import("@/components/dashboard/BoxPlotSection"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

export function S2SDashboard() {
  return (
    <DashboardProvider page="s2s">
      <SidebarMenuProvider>
        <DashboardLayout>
          <KeyMetrics />
          <TimelineChart />
          <BoxPlotSection />
        </DashboardLayout>
      </SidebarMenuProvider>
    </DashboardProvider>
  );
}
