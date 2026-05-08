// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { THIRTY_MINUTES_MS } from "@/lib/config/constants";

/**
 * Normalize a timestamp to the start of its 30-minute bucket.
 * Matches the every-30-minute cron cadence so completion-time stays in the trigger bucket.
 */
export function to30MinuteBucket(timestamp: number): number {
  return Math.floor(timestamp / THIRTY_MINUTES_MS) * THIRTY_MINUTES_MS;
}
