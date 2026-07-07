// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import FacetFilter from "@/components/layout/FacetFilter";

const ModelSidebar: React.FC = () => (
  <div className="hidden lg:block w-72 shrink-0">
    <div className="sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto scrollbar-hide px-4 py-3">
      <FacetFilter />
    </div>
  </div>
);

export default ModelSidebar;
