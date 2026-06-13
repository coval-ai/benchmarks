// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { normalizeModelName } from "@/lib/utils/formatters";
import { useDashboard } from "@/contexts/DashboardContext";
import { getModelColor } from "@/lib/utils/colors";

// Keep the links pinned in place until the footer is about to collide with the
// actual bottom of the link list, then push the sidebar up so the filters slide
// up under the top nav — as if the footer were pushing them off-screen. We
// measure where the links really end (not the full-height container) so a short
// list doesn't get pushed early by the empty space beneath it. The cap keeps a
// long, scrolling list from being shoved before it reaches the viewport floor.
const SIDEBAR_BOTTOM_GAP = 16;
// Gap left between the links and the footer so the links never touch it.
const LINKS_GAP = 24;

const ModelSidebar: React.FC = () => {
  const {
    normalizeProviderName,
    modelsByProvider,
    selectedModels,
    toggleModelSelection: onToggleModelSelection,
  } = useDashboard();

  const containerRef = React.useRef<HTMLDivElement>(null);
  const linksRef = React.useRef<HTMLDivElement>(null);
  const pushUpRef = React.useRef(0);

  React.useEffect(() => {
    // Drive the transform straight on the DOM node so it tracks the scroll in
    // the same frame — going through React state would lag a frame behind.
    const update = () => {
      const container = containerRef.current;
      if (!container) return;
      const footer = document.querySelector("footer");
      const links = linksRef.current;
      if (!footer || !links) {
        pushUpRef.current = 0;
        container.style.transform = "translateY(0px)";
        return;
      }
      const footerTop = footer.getBoundingClientRect().top;
      // Bottom of the links in their un-pushed position: subtract out the
      // transform we're currently applying so the reference point stays stable.
      const naturalLinksBottom =
        links.getBoundingClientRect().bottom + pushUpRef.current;
      const maxBottom = window.innerHeight - SIDEBAR_BOTTOM_GAP;
      const linksBottom = Math.min(naturalLinksBottom, maxBottom);

      const next = Math.max(0, linksBottom + LINKS_GAP - footerTop);
      pushUpRef.current = next;
      container.style.transform = `translateY(-${next}px)`;
    };

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="hidden lg:flex flex-col fixed left-4 top-20 bottom-4 z-10 py-3 w-64"
    >
      {/* Scrollable content area */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        {/* Model Selection Content */}
        <div ref={linksRef} className="space-y-2">
          <div>
            {Object.entries(modelsByProvider).map(([provider, models]) => (
              <div key={provider} className="mt-4 first:mt-0">
                <div className="text-text-primary pt-1.5 pb-0.5 px-2 text-sm font-bold">
                  {normalizeProviderName(provider)}
                </div>
                <div className="space-y-0">
                  {models.map((model) => {
                    const checked = selectedModels.includes(model);
                    const modelColor = getModelColor(model);
                    return (
                      <label
                        key={model}
                        className="flex items-center gap-2 py-1.5 px-2 rounded-lg text-xs cursor-pointer text-text-tertiary"
                      >
                        <span className="relative inline-flex h-3.5 w-3.5 shrink-0">
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => onToggleModelSelection(model)}
                            aria-label={`${
                              checked ? "Deselect" : "Select"
                            } ${model} model`}
                            className="peer h-3.5 w-3.5 shrink-0 cursor-pointer appearance-none rounded-[3px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-text-tertiary/40 focus-visible:ring-offset-1 focus-visible:ring-offset-background"
                            style={{ backgroundColor: checked ? modelColor : "#d4d2cc" }}
                          />
                          <svg
                            aria-hidden
                            viewBox="0 0 14 14"
                            fill="none"
                            stroke="white"
                            strokeWidth={1.5}
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="pointer-events-none absolute inset-0 hidden h-3.5 w-3.5 peer-checked:block"
                          >
                            <path d="M3.5 7.5l2.5 2.5 4.5-5" />
                          </svg>
                        </span>
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
