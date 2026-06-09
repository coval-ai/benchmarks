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
import TimeWindowToggle from "@/components/shared/TimeWindowToggle";

const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const {
    loading,
    loadError,
    benchmarkTitle,
    timeWindow,
    changeTimeWindow,
    windowDataStale,
  } = useDashboard();
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
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background text-text-primary">
      <DashboardHeader />
      <MobileModelSheet />
      <ModelSidebar />

      {/* Content column — centered on the page (under the centered tabs). The
          18rem side gutters keep its left edge clear of the fixed sidebar
          (17rem wide) with a 1rem gap. The footer stays full-width.
          w-full is load-bearing below lg: mx-auto disables the flex
          cross-axis stretch, and the column would otherwise size to its
          content and overflow the viewport. */}
      <div className="relative z-10 flex-1 transition-all duration-300 pt-20 px-3 py-8 sm:px-8 pb-24 lg:pb-8 overflow-x-hidden w-full mx-auto lg:w-[calc(100vw-36rem)]">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-2xl font-bold tracking-tight text-text-primary">
            {benchmarkTitle}
          </h1>
          <TimeWindowToggle
            value={timeWindow}
            onChange={changeTimeWindow}
            className="ml-auto"
          />
        </div>

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

      {!loading && <DashboardFooter />}
    </div>
  );
};

export default DashboardLayout;
