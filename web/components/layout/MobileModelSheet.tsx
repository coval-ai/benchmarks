// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { normalizeModelName } from "@/lib/utils/formatters";
import { useDashboard } from "@/contexts/DashboardContext";

const MobileModelSheet: React.FC = () => {
  const {
    mobileSheetTitle: title,
    normalizeProviderName,
    modelsByProvider,
    selectedModels,
    expandedProviders,
    mobileSheetOpen,
    setMobileSheetOpen: onSetMobileSheetOpen,
    toggleProvider: onToggleProvider,
    toggleModelSelection: onToggleModelSelection,
  } = useDashboard();

  return (
    <div className="lg:hidden">
      {/* Backdrop - only when open */}
      {mobileSheetOpen && (
        <div
          className="fixed inset-0 z-30 bg-surface-backdrop transition-opacity duration-200"
          onClick={() => onSetMobileSheetOpen(false)}
        />
      )}

      {/* Bottom Sheet */}
      <div
        className={`fixed bottom-0 left-0 right-0 z-40 bg-surface-overlay backdrop-blur-xl border-t border-border-primary rounded-t-3xl transition-transform duration-200 ease-out ${
          mobileSheetOpen ? "translate-y-0" : "translate-y-full"
        }`}
        style={{ height: "50vh" }}
      >
        {/* Handle */}
        <div
          className="flex justify-center pt-3 pb-2 cursor-pointer"
          onClick={() => onSetMobileSheetOpen(!mobileSheetOpen)}
        >
          <div className="w-12 h-1 bg-text-tertiary rounded-full"></div>
        </div>

        {/* Sheet Content */}
        <div className="px-4 pb-8 h-full overflow-y-auto">
          <div className="text-text-secondary text-sm font-medium mb-4 text-center">
            {title}
          </div>

          {/* Model Selection */}
          <div className="space-y-2">
            {Object.entries(modelsByProvider).map(([provider, models]) => (
              <div key={provider}>
                <button
                  onClick={() => onToggleProvider(provider)}
                  className="flex items-center justify-between w-full text-text-secondary hover:text-text-primary py-1.5 px-2 rounded-lg text-xs transition-all duration-300 hover:bg-hover-bg group"
                >
                  <span className="font-medium">
                    {normalizeProviderName(provider)}
                  </span>
                  <span
                    className={`text-xs transition-all duration-300 ${
                      expandedProviders[provider] ? "rotate-90" : ""
                    }`}
                  >
                    ›
                  </span>
                </button>
                {expandedProviders[provider] && (
                  <div className="ml-3 space-y-1 mt-2">
                    {models.map((model) => (
                      <button
                        key={model}
                        onClick={() => onToggleModelSelection(model)}
                        className={`block text-left py-2 px-3 rounded-lg text-sm transition-all duration-300 w-full ${
                          selectedModels.includes(model)
                            ? "bg-selected-bg text-text-primary shadow-md border border-selected-border"
                            : "text-text-tertiary hover:text-text-secondary hover:bg-hover-bg"
                        }`}
                      >
                        <span className="font-mono text-sm">
                          {normalizeModelName(model)}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Floating Handle - Only when sheet is closed */}
      {!mobileSheetOpen && (
        <div
          className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-40 bg-surface-overlay backdrop-blur-xl border border-border-primary rounded-full px-4 py-2 cursor-pointer shadow-lg"
          onClick={() => onSetMobileSheetOpen(true)}
        >
          <div className="flex items-center space-x-2 text-text-secondary text-sm">
            <span>{selectedModels.length} selected</span>
            <div className="w-6 h-0.5 bg-text-tertiary rounded-full"></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MobileModelSheet;
