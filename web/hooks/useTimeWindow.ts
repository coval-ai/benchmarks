// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useState } from "react";
import { capturePostHogEvent } from "@/lib/posthog/client";
import {
  POSTHOG_EVENTS,
  type PostHogMode,
  type PostHogSurface,
} from "@/lib/posthog/events";
import type { TimeWindow } from "@/lib/config/timeWindows";

export function useTimeWindow(surface: PostHogSurface, mode?: PostHogMode) {
  const [timeWindow, setTimeWindow] = useState<TimeWindow>("7d");

  const changeTimeWindow = useCallback(
    (next: TimeWindow) => {
      if (next === timeWindow) return;
      capturePostHogEvent(POSTHOG_EVENTS.dashboardTimeWindowChanged, {
        surface,
        ...(mode ? { mode } : {}),
        from: timeWindow,
        to: next
      });
      setTimeWindow(next);
    },
    [timeWindow, surface, mode]
  );

  return { timeWindow, changeTimeWindow };
}
