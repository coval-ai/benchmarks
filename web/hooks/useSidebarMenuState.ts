// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useState } from "react";

export function useSidebarMenuState() {
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);
  const [openFacetGroups, setOpenFacetGroups] = useState<Record<string, boolean>>({});
  return { mobileSheetOpen, setMobileSheetOpen, openFacetGroups, setOpenFacetGroups };
}
