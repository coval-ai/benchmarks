// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import dynamic from "next/dynamic";
import { DashboardProvider } from "@/contexts/DashboardContext";
import { SidebarMenuProvider } from "@/contexts/SidebarMenuContext";
import DashboardLayout from "@/components/layout/DashboardLayout";
import KeyMetrics from "@/components/dashboard/KeyMetrics";
import { ChartSkeleton } from "@/components/dashboard/DashboardSkeleton";

// Lazy-load heavy chart components — D3 and Recharts do not support SSR.
// Each renders a chart-sized skeleton while its chunk loads so the page keeps
// its full height and the footer never flashes up into view during the gap.
const TimelineChart = dynamic(
  () => import("@/components/visualizations/TimelineChart"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const BoxPlotSection = dynamic(
  () => import("@/components/dashboard/BoxPlotSection"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const LatencyAccuracySection = dynamic(
  () => import("@/components/dashboard/LatencyAccuracySection"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const AccuracyBarSection = dynamic(
  () => import("@/components/dashboard/AccuracyBarSection"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const HeatmapSection = dynamic(
  () => import("@/components/dashboard/HeatmapSection"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

export function STTDashboard() {
  return (
    <DashboardProvider page="stt">
      <SidebarMenuProvider>
        <DashboardLayout>
          <KeyMetrics />
          <TimelineChart />
          <BoxPlotSection />
          <AccuracyBarSection />
          <LatencyAccuracySection />
          <HeatmapSection />
        </DashboardLayout>
      </SidebarMenuProvider>
    </DashboardProvider>
  );
}
