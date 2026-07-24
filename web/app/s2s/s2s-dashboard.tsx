// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import dynamic from "next/dynamic";
import { DashboardProvider } from "@/contexts/DashboardContext";
import { SidebarMenuProvider } from "@/contexts/SidebarMenuContext";
import DashboardLayout from "@/components/layout/DashboardLayout";
import { ChartSkeleton } from "@/components/dashboard/DashboardSkeleton";
import { S2STopRow } from "./components/S2STopRow";

// S2S plots V2V latency plus instruction adherence (via QualityBarSection, the
// shared quality-bar section — WER for STT/TTS, instruction here). The WER-only
// LatencyAccuracySection/WerRadar are omitted; the model comparison table
// renders WER-free (it hides the column when rows lack it).
const TimelineChart = dynamic(
  () => import("@/components/visualizations/TimelineChart"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const BoxPlotSection = dynamic(
  () => import("@/components/dashboard/BoxPlotSection"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

const QualityBarSection = dynamic(
  () => import("@/components/dashboard/QualityBarSection"),
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
          <S2STopRow />
          <TimelineChart />
          <BoxPlotSection />
          <QualityBarSection />
          <ModelComparisonSection />
        </DashboardLayout>
      </SidebarMenuProvider>
    </DashboardProvider>
  );
}
