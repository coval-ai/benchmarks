// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useEffect, useRef } from "react";
import { useActiveTab } from "@/hooks/useActiveTab";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS, type DashboardChartId } from "@/lib/posthog/events";

export function useChartHoverTracking(chart: DashboardChartId): () => void {
  const mode = useActiveTab();
  const firedRef = useRef(false);

  useEffect(() => {
    firedRef.current = false;
  }, [mode, chart]);

  return useCallback(() => {
    if (firedRef.current) return;
    firedRef.current = true;
    capturePostHogEvent(POSTHOG_EVENTS.dashboardChartHovered, {
      surface: `${mode}_dashboard`,
      mode,
      chart
    });
  }, [mode, chart]);
}
