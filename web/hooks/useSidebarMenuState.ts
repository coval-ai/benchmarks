// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import type { ModelsByProvider } from "@/types/benchmark.types";

export function useSidebarMenuState(modelsByProvider: ModelsByProvider) {
  const [expandedProviders, setExpandedProviders] = useState<{
    [key: string]: boolean;
  }>({});
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);

  // Expand every provider group once, the first time models arrive.
  const hasExpandedRef = useRef(false);
  useEffect(() => {
    if (hasExpandedRef.current) return;
    const providers = Object.keys(modelsByProvider);
    if (providers.length === 0) return;
    const expanded: Record<string, boolean> = {};
    providers.forEach((provider) => {
      expanded[provider] = true;
    });
    setExpandedProviders(expanded);
    hasExpandedRef.current = true;
  }, [modelsByProvider]);

  const toggleProvider = useCallback((provider: string) => {
    setExpandedProviders((prev) => ({
      ...prev,
      [provider]: !prev[provider],
    }));
  }, []);

  return useMemo(
    () => ({
      expandedProviders,
      mobileSheetOpen,
      setMobileSheetOpen,
      toggleProvider,
    }),
    [expandedProviders, mobileSheetOpen, toggleProvider]
  );
}
