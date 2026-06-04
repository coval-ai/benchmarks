// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { createContext, useContext } from "react";
import { useSidebarMenuState } from "@/hooks/useSidebarMenuState";
import { useDashboard } from "@/contexts/DashboardContext";

type SidebarMenuContextType = ReturnType<typeof useSidebarMenuState>;

const SidebarMenuContext = createContext<SidebarMenuContextType | null>(null);

export function SidebarMenuProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { modelsByProvider } = useDashboard();
  const value = useSidebarMenuState(modelsByProvider);
  return (
    <SidebarMenuContext.Provider value={value}>
      {children}
    </SidebarMenuContext.Provider>
  );
}

export function useSidebarMenu(): SidebarMenuContextType {
  const context = useContext(SidebarMenuContext);
  if (!context) {
    throw new Error(
      "useSidebarMenu must be used within a SidebarMenuProvider"
    );
  }
  return context;
}
