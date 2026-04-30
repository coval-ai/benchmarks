"use client";

import dynamic from "next/dynamic";
import { DashboardProvider } from "@/contexts/DashboardContext";
import DashboardLayout from "@/components/layout/DashboardLayout";
import PageHeader from "@/components/dashboard/PageHeader";
import KeyMetrics from "@/components/dashboard/KeyMetrics";
import DashboardFooter from "@/components/dashboard/DashboardFooter";

// Lazy-load heavy chart components — D3 and Recharts do not support SSR
const TimelineChart = dynamic(
  () => import("@/components/visualizations/TimelineChart"),
  { ssr: false }
);

const ViolinSection = dynamic(
  () => import("@/components/dashboard/ViolinSection"),
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

export function TTSDashboard() {
  return (
    <DashboardProvider page="tts">
      <DashboardLayout>
        <PageHeader />
        <KeyMetrics />
        <TimelineChart />
        <ViolinSection />
        <LatencyAccuracySection />
        <AccuracyBarSection />
        <HeatmapSection />
        <DashboardFooter />
      </DashboardLayout>
    </DashboardProvider>
  );
}
