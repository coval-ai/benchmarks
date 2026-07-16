// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";

const WerBarViewToggle: React.FC = () => {
  const { werBarView, changeWerBarView, availableWerBarViews } = useDashboard();

  if (availableWerBarViews.length <= 1) return null;

  return (
    <div className="mb-4 inline-flex gap-0.5 rounded-lg bg-surface-toggle-inactive p-0.5">
      {availableWerBarViews.map((view) => (
        <button
          key={view.key}
          type="button"
          onClick={() => changeWerBarView(view.key)}
          className={
            "rounded-md px-4 py-3 text-sm sm:px-3 sm:py-1 sm:text-xs font-medium transition-colors " +
            (werBarView === view.key
              ? "bg-surface-primary text-text-primary shadow-sm"
              : "text-text-secondary hover:text-text-primary")
          }
        >
          {view.label}
        </button>
      ))}
    </div>
  );
};

export default WerBarViewToggle;
