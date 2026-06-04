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
    toggleModelSelection: onToggleModelSelection,
  } = useDashboard();

  return (
    <div className="hidden lg:flex flex-col fixed left-4 top-28 bottom-4 z-10 py-3 w-64">
      {/* Scrollable content area */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        {/* Model Selection Content */}
        <div className="space-y-2">
          <div>
            <div className="text-text-secondary text-xs font-medium mb-3 px-2">
              {title}
            </div>

            {Object.entries(modelsByProvider).map(([provider, models]) => (
              <div key={provider} className="mt-4 first:mt-0">
                <div className="text-text-primary pt-1.5 pb-0.5 px-2 text-sm font-bold">
                  {normalizeProviderName(provider)}
                </div>
                <div className="space-y-0">
                  {models.map((model) => {
                    const checked = selectedModels.includes(model);
                    return (
                      <label
                        key={model}
                        className="flex items-center gap-2 py-1.5 px-2 rounded-lg text-xs cursor-pointer text-text-tertiary"
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => onToggleModelSelection(model)}
                          aria-label={`${
                            checked ? "Deselect" : "Select"
                          } ${model} model`}
                          className="h-3.5 w-3.5 shrink-0 rounded accent-text-primary cursor-pointer"
                        />
                        <span
                          className={`truncate text-xs leading-tight ${
                            checked ? "text-text-primary" : ""
                          }`}
                        >
                          {normalizeModelName(model)}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelSidebar;
