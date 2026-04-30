// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useState, useCallback } from "react";

interface BarDataPoint {
  model: string;
  averageWER: number;
  provider: string;
}

export const useBarInteraction = () => {
  const [clickedWERBars, setClickedWERBars] = useState<Set<string>>(new Set());

  const handleWERBarClick = useCallback((data: BarDataPoint | null) => {
    if (!data || !data.model) return;

    const modelName = data.model;
    setClickedWERBars((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(modelName)) {
        newSet.delete(modelName);
      } else {
        newSet.add(modelName);
      }
      return newSet;
    });
  }, []);

  return {
    clickedWERBars,
    handleWERBarClick
  };
};
