// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import MobileModelSheet from "@/components/layout/MobileModelSheet";
import ModelSidebar from "@/components/layout/ModelSidebar";

const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { sidebarCollapsed, loading } = useDashboard();

  // Offset for the content column so the fixed sidebar doesn't overlap it.
  // The footer is intentionally left full-width (spanning under the sidebar).
  const columnOffset = sidebarCollapsed ? "lg:ml-20" : "lg:ml-72";

  return (
    <div className="min-h-screen bg-background text-text-primary">
      <DashboardHeader />
      <MobileModelSheet />
      <ModelSidebar />

      <div
        className={`transition-all duration-300 pt-28 p-8 pb-24 lg:pb-8 overflow-x-hidden ${columnOffset} ${
          sidebarCollapsed ? "lg:w-[calc(100vw-5rem)]" : "lg:w-[calc(100vw-18rem)]"
        }`}
      >
        {loading && (
          <div className="-mx-8 w-screen flex flex-col items-center justify-center min-h-[70vh] lg:-translate-x-9 bg-background text-text-primary">
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

      {!loading && <DashboardFooter />}
    </div>
  );
};

export default DashboardLayout;
