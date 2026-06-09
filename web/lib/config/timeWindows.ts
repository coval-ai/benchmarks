// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { TWENTY_FOUR_HOURS_MS } from "@/lib/config/constants";

export const TIME_WINDOWS = ["24h", "7d", "30d"] as const;
export type TimeWindow = (typeof TIME_WINDOWS)[number];

// Values mirror the API's window literals; "1d" is display-only.
export const WINDOW_LABELS: Record<TimeWindow, string> = {
  "24h": "1d",
  "7d": "7d",
  "30d": "30d",
};

export const WINDOW_MS: Record<TimeWindow, number> = {
  "24h": TWENTY_FOUR_HOURS_MS,
  "7d": 7 * TWENTY_FOUR_HOURS_MS,
  "30d": 30 * TWENTY_FOUR_HOURS_MS,
};
