import { FIFTEEN_MINUTES_MS } from "@/lib/config/constants";

/**
 * Normalize a timestamp to the start of its 15-minute bucket.
 */
export function to15MinuteBucket(timestamp: number): number {
  return Math.floor(timestamp / FIFTEEN_MINUTES_MS) * FIFTEEN_MINUTES_MS;
}
