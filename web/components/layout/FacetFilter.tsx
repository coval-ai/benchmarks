// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState } from "react";
import { ChevronDown } from "lucide-react";
import { useDashboard } from "@/contexts/DashboardContext";
import { getReadableTextColor } from "@/lib/utils/colors";

const COLLAPSE_THRESHOLD = 8;

const FacetFilter: React.FC = () => {
  const { facetGroups, toggleFacet, clearFacets, hasActiveFacets } =
    useDashboard();
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  if (facetGroups.length === 0) return null;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between px-2 pb-2">
        <span className="text-sm font-bold text-text-primary">Filters</span>
        {hasActiveFacets && (
          <button
            type="button"
            onClick={clearFacets}
            className="text-xs text-text-tertiary transition-colors hover:text-text-primary"
          >
            Clear
          </button>
        )}
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
              className="flex min-h-11 w-full items-center justify-between gap-2 py-2 text-[11px] font-semibold uppercase tracking-wide text-text-tertiary transition-colors hover:text-text-primary lg:min-h-0"
            >
              <span>{group.label}</span>
              <span className="flex items-center gap-1.5">
                {activeCount > 0 && (
                  <span className="rounded-full bg-surface-toggle-active px-1.5 py-px text-[10px] tabular-nums text-text-on-toggle-active">
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
                  const fill = option.color ?? null;
                  const fg = fill ? getReadableTextColor(fill) : null;
                  const ringed = option.active || option.implied;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      aria-pressed={option.active}
                      onClick={() => toggleFacet(group.category, option.value)}
                      style={
                        fill
                          ? {
                              backgroundColor: fill,
                              color: fg!,
                              borderColor: "transparent",
                              boxShadow: ringed
                                ? "0 0 0 2px var(--color-surface-primary), 0 0 0 3.5px var(--color-text-primary)"
                                : undefined,
                            }
                          : undefined
                      }
                      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors focus-visible:outline-none lg:px-2.5 lg:py-1 focus-visible:ring-1 focus-visible:ring-text-tertiary/40 ${
                        fill
                          ? ""
                          : option.active
                            ? "border-transparent bg-surface-toggle-active text-text-on-toggle-active"
                            : "border-border-primary text-text-secondary hover:border-text-tertiary/50 hover:text-text-primary"
                      }`}
                    >
                      <span>{option.label}</span>
                      <span
                        style={{
                          minWidth: `${String(option.maxCount).length}ch`,
                        }}
                        className={`inline-block text-center tabular-nums ${
                          fill
                            ? "opacity-70"
                            : option.active
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
