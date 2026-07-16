// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import FacetFilter from "@/components/layout/FacetFilter";

// The margin animates across the lg breakpoint so the filters slide in and
// the content column reflows with them, instead of teleporting 18rem when a
// window drag crosses 50% of the screen. Within a regime the margin is
// constant, so plain resizes never fire the transition. Sliding via margin
// (clipped by the page root's overflow-x-clip) rather than width +
// overflow-hidden keeps this ancestor scroll-free, which position: sticky
// needs to track the viewport.
const ModelSidebar: React.FC = () => (
  <div className="w-72 -ml-72 lg:ml-0 shrink-0 invisible lg:visible transition-[margin-left,visibility] duration-300">
    <div className="sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto scrollbar-hide px-4 py-3">
      <FacetFilter />
    </div>
  </div>
);

export default ModelSidebar;
