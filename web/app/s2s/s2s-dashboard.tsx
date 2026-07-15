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
// (AccuracyBarSection, LatencyAccuracySection) are omitted; the model
// comparison table renders WER-free (it hides the column when rows lack it).
const TimelineChart = dynamic(
  () => import("@/components/visualizations/TimelineChart"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const BoxPlotSection = dynamic(
  () => import("@/components/dashboard/BoxPlotSection"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const ModelComparisonSection = dynamic(
  () => import("@/components/dashboard/ModelComparisonSection"),
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
          <ModelComparisonSection />
        </DashboardLayout>
      </SidebarMenuProvider>
    </DashboardProvider>
  );
}
