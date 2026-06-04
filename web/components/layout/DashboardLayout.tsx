// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import MobileModelSheet from "@/components/layout/MobileModelSheet";
import ModelSidebar from "@/components/layout/ModelSidebar";
import DashboardSkeleton from "@/components/dashboard/DashboardSkeleton";

const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { loading, benchmarkTitle } = useDashboard();

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-text-primary">
      <DashboardHeader />
      <MobileModelSheet />
      <ModelSidebar />

      {/* Content column — centered on the page (under the centered tabs). The
          18rem side gutters keep its left edge clear of the fixed sidebar
          (17rem wide) with a 1rem gap. The footer stays full-width. */}
      <div className="relative z-10 transition-all duration-300 pt-20 p-8 pb-24 lg:pb-8 overflow-x-hidden mx-auto lg:w-[calc(100vw-36rem)]">
        <h1 className="mb-6 text-2xl font-bold tracking-tight text-text-primary">
          {benchmarkTitle}
        </h1>

        {loading ? <DashboardSkeleton /> : children}
      </div>

      {!loading && <DashboardFooter />}
    </div>
  );
};

export default DashboardLayout;
