// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useState } from "react";

export function useSidebarMenuState() {
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);
  return { mobileSheetOpen, setMobileSheetOpen };
}
