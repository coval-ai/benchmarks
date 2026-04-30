// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { createContext, useContext } from "react";
import { useDashboardState } from "@/hooks/useDashboardState";

type DashboardContextType = ReturnType<typeof useDashboardState>;

const DashboardContext = createContext<DashboardContextType | null>(null);

export function DashboardProvider({
  page,
  children,
}: {
  page: "tts" | "stt";
  children: React.ReactNode;
}) {
  const state = useDashboardState(page);
  return (
    <DashboardContext.Provider value={state}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard(): DashboardContextType {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error("useDashboard must be used within a DashboardProvider");
  }
  return context;
}
