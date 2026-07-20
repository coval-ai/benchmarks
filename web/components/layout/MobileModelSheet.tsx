// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import { useSidebarMenu } from "@/contexts/SidebarMenuContext";
import FacetFilter from "@/components/layout/FacetFilter";
import { CymaticLoader } from "@/components/shared/CymaticLoader";

const MobileModelSheet: React.FC = () => {
  const {
    mobileSheetTitle: title,
    hasActiveFacets,
    windowDataStale,
  } = useDashboard();
  const {
    mobileSheetOpen,
    setMobileSheetOpen: onSetMobileSheetOpen,
  } = useSidebarMenu();

  return (
    <div className="lg:hidden">
      {/* Backdrop - only when open */}
      {mobileSheetOpen && (
        <button
          type="button"
          aria-label="Close filters"
          className="fixed inset-0 z-30 bg-surface-backdrop transition-opacity duration-200"
          onClick={() => onSetMobileSheetOpen(false)}
        />
      )}

      {/* Bottom Sheet */}
      <div
        aria-hidden={!mobileSheetOpen}
        inert={!mobileSheetOpen}
        className={`fixed bottom-0 left-0 right-0 z-40 bg-surface-overlay backdrop-blur-xl border-t border-border-primary rounded-t-3xl transition-transform duration-200 ease-out ${
          mobileSheetOpen ? "translate-y-0" : "translate-y-full"
        }`}
        style={{ height: "50vh" }}
      >
        {/* Handle */}
        <button
          type="button"
          aria-label="Close filters"
          className="flex min-h-11 w-full items-center justify-center"
          onClick={() => onSetMobileSheetOpen(!mobileSheetOpen)}
        >
          <div className="w-12 h-1 bg-text-tertiary rounded-full"></div>
        </button>

        {/* Sheet Content */}
        <div className="px-4 pb-8 h-full overflow-y-auto">
          <div className="text-text-secondary text-sm font-medium mb-4 text-center">
            {title}
          </div>

          <FacetFilter />
        </div>
      </div>

      {/* Floating Handle - Only when sheet is closed */}
      {!mobileSheetOpen && (
        <button
          type="button"
          aria-label="Open filters"
          className="fixed bottom-4 left-1/2 z-40 min-h-11 -translate-x-1/2 rounded-full border border-border-primary bg-surface-overlay px-4 py-2 shadow-lg backdrop-blur-xl"
          onClick={() => onSetMobileSheetOpen(true)}
        >
          <div className="flex items-center space-x-2 text-text-secondary text-sm">
            <span>Filters</span>
            {hasActiveFacets && (
              <span
                className="h-2 w-2 rounded-full bg-surface-toggle-active"
                aria-label="Filters active"
              />
            )}
            {windowDataStale ? (
              <CymaticLoader size={20} animated className="text-text-primary" />
            ) : (
              <div className="w-6 h-0.5 bg-text-tertiary rounded-full"></div>
            )}
          </div>
        </button>
      )}
    </div>
  );
};

export default MobileModelSheet;
