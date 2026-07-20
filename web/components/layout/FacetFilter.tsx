// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { ChevronDown } from "lucide-react";
import { useDashboard } from "@/contexts/DashboardContext";
import { useSidebarMenu } from "@/contexts/SidebarMenuContext";
import TimeWindowToggle from "@/components/shared/TimeWindowToggle";
import { CymaticLoader } from "@/components/shared/CymaticLoader";

const COLLAPSE_THRESHOLD = 8;

const FacetFilter: React.FC = () => {
  const {
    facetGroups,
    toggleFacet,
    clearFacets,
    hasActiveFacets,
    timeWindow,
    changeTimeWindow,
    windowDataStale,
  } = useDashboard();
  const { openFacetGroups: openGroups, setOpenFacetGroups: setOpenGroups } =
    useSidebarMenu();

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between px-2 pb-2">
        <span className="text-sm font-bold text-text-primary">Filters</span>
        {hasActiveFacets && (
          <button
            type="button"
            onClick={clearFacets}
            className="min-h-11 px-2 text-xs text-text-tertiary transition-colors hover:text-text-primary lg:min-h-0"
          >
            Clear
          </button>
        )}
      </div>

      <div
        role="group"
        aria-labelledby="facet-time-range"
        className="px-2 py-1"
      >
        <div className="flex min-h-11 items-center gap-2 py-2 lg:min-h-0">
          <span
            id="facet-time-range"
            className="font-mono text-[13px] text-text-tertiary"
          >
            Time range
          </span>
          <CymaticLoader
            size={20}
            animated={windowDataStale}
            className={`text-text-primary transition-opacity duration-300 ${
              windowDataStale ? "opacity-100" : "opacity-0"
            }`}
          />
        </div>
        <TimeWindowToggle
          value={timeWindow}
          onChange={changeTimeWindow}
          loading={windowDataStale}
          className="mb-2 w-full"
        />
      </div>

      {facetGroups.map((group) => {
        const activeCount = group.options.filter((o) => o.active).length;
        const isOpen =
          openGroups[group.category] ??
          group.options.length <= COLLAPSE_THRESHOLD;
        const visibleOptions = isOpen
          ? group.options
          : group.options.filter((o) => o.active || o.implied);
        return (
          <div
            key={group.category}
            role="group"
            aria-labelledby={`facet-${group.category}`}
            className="border-t border-border-primary px-2 py-1"
          >
            <button
              type="button"
              id={`facet-${group.category}`}
              aria-expanded={isOpen}
              onClick={() =>
                setOpenGroups((prev) => ({
                  ...prev,
                  [group.category]: !isOpen,
                }))
              }
              className="flex min-h-11 w-full items-center justify-between gap-2 py-2 font-mono text-[13px] text-text-tertiary transition-colors hover:text-text-primary lg:min-h-0"
            >
              <span>{group.label}</span>
              <span className="flex items-center gap-1.5">
                {activeCount > 0 && (
                  <span className="rounded-full bg-surface-toggle-active px-1.5 py-px font-mono text-[10px] tabular-nums text-text-on-toggle-active">
                    {activeCount}
                  </span>
                )}
                <ChevronDown
                  size={14}
                  aria-hidden="true"
                  className={`transition-transform ${isOpen ? "rotate-180" : ""}`}
                />
              </span>
            </button>
            {visibleOptions.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pb-2">
                {visibleOptions.map((option) => {
                  return (
                    <button
                      key={option.value}
                      type="button"
                      aria-pressed={option.active}
                      onClick={() => toggleFacet(group.category, option.value)}
                      className={`inline-flex min-h-11 items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors focus-visible:outline-none lg:min-h-0 lg:px-2.5 lg:py-1 focus-visible:ring-1 focus-visible:ring-text-tertiary/40 ${
                        option.active
                          ? "border-transparent bg-surface-toggle-active text-text-on-toggle-active"
                          : option.implied
                            ? "border-text-primary text-text-primary"
                            : "border-border-primary text-text-secondary hover:border-text-tertiary/50 hover:text-text-primary"
                      }`}
                    >
                      <span>{option.label}</span>
                      <span
                        style={{
                          minWidth: `${String(option.maxCount).length}ch`,
                        }}
                        className={`inline-block text-center font-mono tabular-nums ${
                          option.active
                            ? "text-text-on-toggle-active/70"
                            : "text-text-tertiary"
                        }`}
                      >
                        {option.count}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default FacetFilter;
