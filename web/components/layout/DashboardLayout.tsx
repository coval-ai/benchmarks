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

const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { loading } = useDashboard();
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
    <div className="min-h-screen bg-background text-text-primary">
      <DashboardHeader />
      <MobileModelSheet />
      <ModelSidebar />

      {/* Below 1440px content fills the space right of the sidebar; from 1440px
          the gutters go symmetric so content centers with the nav and footer. */}
      <div className="pt-28 p-8 pb-24 lg:pb-8 overflow-x-hidden lg:ml-52 min-[90rem]:mr-52">
        <div className="max-w-[1400px] mx-auto">
          {loading && (
            <div className="flex flex-col items-center justify-center min-h-[70vh] bg-background text-text-primary">
              <div className="w-6 h-6 border-[3px] border-spinner-track border-t-spinner-head rounded-full animate-spin" />
              <h1 className="mt-6 text-base tracking-tight">
                Loading benchmarks
              </h1>
            </div>
          )}

          {!loading && (
            <>
              <div className="mb-16"></div>
              {children}
            </>
          )}
        </div>
      </div>

      {!loading && <DashboardFooter />}
    </div>
  );
};

export default DashboardLayout;
