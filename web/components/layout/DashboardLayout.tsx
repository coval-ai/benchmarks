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
  const { loading } = useDashboard();

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
