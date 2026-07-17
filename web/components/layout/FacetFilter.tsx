// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState } from "react";
import { useDashboard } from "@/contexts/DashboardContext";

const FacetFilter: React.FC = () => {
  const { facetGroups, toggleFacet, clearFacets, hasActiveFacets } = useDashboard();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

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

      {facetGroups.map((group) => {
        const isOpen = expanded[group.category] ?? false;
        const selectedCount = group.options.filter((o) => o.active).length;
        return (
          <div
            key={group.category}
            role="group"
            aria-labelledby={`facet-${group.category}`}
            className="px-2"
          >
            <button
              type="button"
              id={`facet-${group.category}`}
              aria-expanded={isOpen}
              onClick={() =>
                setExpanded((prev) => ({ ...prev, [group.category]: !isOpen }))
              }
              className="mb-1.5 flex w-full items-center justify-between text-[11px] font-semibold uppercase tracking-wide text-text-tertiary transition-colors hover:text-text-primary"
            >
              <span className="flex items-center gap-1.5">
                {group.label}
                {selectedCount > 0 && (
                  <span className="rounded-full bg-surface-toggle-active px-1.5 text-[10px] tabular-nums text-text-on-toggle-active">
                    {selectedCount}
                  </span>
                )}
              </span>
              <svg
                viewBox="0 0 12 12"
                aria-hidden="true"
                className={`h-3 w-3 transition-transform ${isOpen ? "rotate-180" : ""}`}
              >
                <path
                  d="M2.5 4.5 6 8l3.5-3.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
            {isOpen && (
              <div className="flex flex-wrap gap-1.5">
                {group.options.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    aria-pressed={option.active}
                    onClick={() => toggleFacet(group.category, option.value)}
                    style={
                      option.implied && !option.active
                        ? {
                            boxShadow:
                              "0 0 0 2px var(--color-surface-primary), 0 0 0 3.5px var(--color-text-primary)",
                          }
                        : undefined
                    }
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 ${
                      option.active
                        ? "border-transparent bg-surface-toggle-active text-text-on-toggle-active"
                        : "border-border-primary text-text-secondary hover:border-text-tertiary/50 hover:text-text-primary"
                    }`}
                  >
                    <span>{option.label}</span>
                    <span
                      style={{ minWidth: `${String(option.maxCount).length}ch` }}
                      className={`inline-block text-center tabular-nums ${
                        option.active ? "text-text-on-toggle-active/70" : "text-text-tertiary"
                      }`}
                    >
                      {option.count}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default FacetFilter;
