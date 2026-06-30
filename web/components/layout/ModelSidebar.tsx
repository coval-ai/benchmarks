// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import FacetFilter from "@/components/layout/FacetFilter";

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
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        <div ref={linksRef}>
          <FacetFilter />
        </div>
      </div>
    </div>
  );
};

export default ModelSidebar;
