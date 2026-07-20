// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useRef } from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import { useActiveTab } from "@/hooks/useActiveTab";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";
import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import MobileModelSheet from "@/components/layout/MobileModelSheet";
import ModelSidebar from "@/components/layout/ModelSidebar";
import DashboardSkeleton from "@/components/dashboard/DashboardSkeleton";

const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { loading, loadError, benchmarkTitle, windowDataStale } =
    useDashboard();
  const mode = useActiveTab();
  const firedDepthsRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    firedDepthsRef.current.clear();
    const thresholds = [25, 50, 75, 100];
    const onScroll = () => {
      const scrollable =
        document.documentElement.scrollHeight - window.innerHeight;
      if (scrollable <= 0) return;
      const pct = Math.min(100, Math.round((window.scrollY / scrollable) * 100));
      for (const t of thresholds) {
        if (pct >= t && !firedDepthsRef.current.has(t)) {
          firedDepthsRef.current.add(t);
          capturePostHogEvent(POSTHOG_EVENTS.dashboardScrollDepth, {
            surface: `${mode}_dashboard`,
            mode,
            depth_pct: t
          });
        }
      }
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [mode]);

  return (
    <div className="relative flex min-h-screen flex-col overflow-x-clip bg-background text-text-primary">
      <DashboardHeader />
      <MobileModelSheet />

      {/* Content column fills whatever the sticky sidebar leaves — it tracks
          window resizes instantly (no width transition, which would drag the
          charts through a 300ms animation on every resize). min-w-0 is
          load-bearing: the column would otherwise size to its content and
          overflow the viewport. The footer sits below this row, so it pushes
          the sticky sidebar up natively. */}
      <div className="relative z-10 flex flex-1">
        <ModelSidebar />
        <div className="pt-20 px-3 py-8 sm:px-8 pb-24 lg:pb-8 overflow-x-hidden flex-1 min-w-0">
          <h1 className="mb-6 text-2xl font-bold tracking-tight text-text-primary">
            {benchmarkTitle}
          </h1>

          <div
            className={`transition-opacity duration-200 ${
              windowDataStale ? "opacity-60" : ""
            }`}
          >
            {loading ? (
              <DashboardSkeleton />
            ) : loadError ? (
              <div className="py-24 text-center text-sm text-text-tertiary">
                Couldn&rsquo;t load benchmark results. Try another time window or
                refresh the page.
              </div>
            ) : (
              children
            )}
          </div>
        </div>
      </div>

      {!loading && <DashboardFooter />}
    </div>
  );
};

export default DashboardLayout;
