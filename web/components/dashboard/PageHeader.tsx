// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";

const PageHeader: React.FC = () => {
  const { pageTitle, pageSubtitle } = useDashboard();

  return (
    <div className="mb-16 text-center">
      <h1 className="text-5xl md:text-6xl font-bold tracking-tight leading-[1.05] text-text-primary mb-6">
        {pageTitle}
      </h1>
      <p className="text-xl md:text-2xl font-light leading-snug text-text-secondary mb-6 max-w-3xl mx-auto">
        {pageSubtitle}
      </p>
    </div>
  );
};

export default PageHeader;
