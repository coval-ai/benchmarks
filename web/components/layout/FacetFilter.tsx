// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import { getReadableTextColor } from "@/lib/utils/colors";

const FacetFilter: React.FC = () => {
  const { facetGroups, toggleFacet, clearFacets, hasActiveFacets } = useDashboard();

  if (facetGroups.length === 0) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between px-2">
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

      {facetGroups.map((group) => (
        <div
          key={group.category}
          role="group"
          aria-labelledby={`facet-${group.category}`}
          className="px-2"
        >
          <div
            id={`facet-${group.category}`}
            className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-text-tertiary"
          >
            {group.label}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {group.options.map((option) => {
              const fill = option.color ?? null;
              const fg = fill ? getReadableTextColor(fill) : null;
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
                          boxShadow: option.active
                            ? "0 0 0 2px var(--color-surface-primary), 0 0 0 3.5px var(--color-text-primary)"
                            : undefined,
                        }
                      : undefined
                  }
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 ${
                    fill
                      ? ""
                      : option.active
                        ? "border-transparent bg-surface-toggle-active text-text-on-toggle-active"
                        : "border-border-primary text-text-secondary hover:border-text-tertiary/50 hover:text-text-primary"
                  }`}
                >
                  <span>{option.label}</span>
                  <span
                    style={{ minWidth: `${String(option.maxCount).length}ch` }}
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
        </div>
      ))}
    </div>
  );
};

export default FacetFilter;
