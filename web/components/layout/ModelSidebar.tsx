// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { normalizeModelName } from "@/lib/utils/formatters";
import { useDashboard } from "@/contexts/DashboardContext";


const ModelSidebar: React.FC = () => {
  const {
    sidebarTitle: title,
    normalizeProviderName,
    modelsByProvider,
    selectedModels,
    expandedProviders,
    sidebarCollapsed,
    toggleSidebar: onToggleSidebar,
    toggleProvider: onToggleProvider,
    toggleModelSelection: onToggleModelSelection,
  } = useDashboard();

  return (
    <div
      className={`hidden lg:flex flex-col fixed left-4 top-20 bottom-4 z-10 bg-surface-secondary backdrop-blur-2xl border border-border-primary rounded-3xl p-3 shadow-2xl transition-all duration-300 ${
        sidebarCollapsed ? "w-16" : "w-64"
      }`}
    >
      {/* Scrollable content area */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
      {/* Toggle Button */}
      <button
        onClick={onToggleSidebar}
        className="w-full mb-4 p-2 hover:bg-hover-bg rounded-xl transition-colors flex items-center justify-center"
        aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
          />
        </svg>
      </button>

      {/* Model Selection Content */}
      {!sidebarCollapsed ? (
        // Expanded state - show full model selection
        <div className="space-y-2">
          <div>
            <div className="text-text-secondary text-xs font-medium mb-3 px-2">
              {title}
            </div>

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
                  <div className="ml-3 space-y-0.5 mt-1 animate-in slide-in-from-top-1 duration-200">
                    {models.map((model) => (
                      <button
                        key={model}
                        onClick={() => onToggleModelSelection(model)}
                        aria-label={`${
                          selectedModels.includes(model)
                            ? "Deselect"
                            : "Select"
                        } ${model} model`}
                        className={`block text-left py-1.5 px-2 rounded-lg text-xs transition-all duration-300 w-full group ${
                          selectedModels.includes(model)
                            ? "bg-selected-bg text-text-primary shadow-md border border-selected-border"
                            : "text-text-tertiary hover:text-text-secondary hover:bg-hover-bg"
                        }`}
                      >
                        <span className="truncate font-mono text-xs leading-tight">
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
      ) : (
        // Collapsed state - show compact indicators
        <div className="space-y-3">
          <div className="text-center">
            <div className="text-xs text-text-secondary">
              {selectedModels.length}
            </div>
            <div className="text-xs text-text-tertiary">models</div>
          </div>
        </div>
      )}
      </div>

    </div>
  );
};

export default ModelSidebar;
