// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

const DAY_MS = 24 * 60 * 60 * 1000;

export const TIME_WINDOWS = ["24h", "7d", "30d"] as const;
export type TimeWindow = (typeof TIME_WINDOWS)[number];

// Values mirror the API's window literals; "1d" is display-only.
export const WINDOW_LABELS: Record<TimeWindow, string> = {
  "24h": "1d",
  "7d": "7d",
  "30d": "30d",
};

export const WINDOW_MS: Record<TimeWindow, number> = {
  "24h": DAY_MS,
  "7d": 7 * DAY_MS,
  "30d": 30 * DAY_MS,
};
