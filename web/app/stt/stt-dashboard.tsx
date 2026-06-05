// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import dynamic from "next/dynamic";
import { DashboardProvider } from "@/contexts/DashboardContext";
import { SidebarMenuProvider } from "@/contexts/SidebarMenuContext";
import DashboardLayout from "@/components/layout/DashboardLayout";
import KeyMetrics from "@/components/dashboard/KeyMetrics";

// Lazy-load heavy chart components — D3 and Recharts do not support SSR
const TimelineChart = dynamic(
  () => import("@/components/visualizations/TimelineChart"),
  { ssr: false }
);

const BoxPlotSection = dynamic(
  () => import("@/components/dashboard/BoxPlotSection"),
  { ssr: false }
);

const LatencyAccuracySection = dynamic(
  () => import("@/components/dashboard/LatencyAccuracySection"),
  { ssr: false }
);

const AccuracyBarSection = dynamic(
  () => import("@/components/dashboard/AccuracyBarSection"),
  { ssr: false }
);

const HeatmapSection = dynamic(
  () => import("@/components/dashboard/HeatmapSection"),
  { ssr: false }
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
