// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * Benchmarking-team access to early-access models.
 *
 * Landing on any page with `?internal=<key>` stores the key in localStorage
 * and strips it from the URL; every subsequent API request sends it as
 * `X-Internal-Key`, which unlocks early-access models server-side.
 * `?internal=` (empty value) clears the stored key.
 */

const STORAGE_KEY = "coval-internal-key";
const QUERY_PARAM = "internal";

export function getInternalKey(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

/** Store the key from ?internal=<key>. Safe to call during render (idempotent). */
export function captureInternalKeyFromUrl(): void {
  const key = new URL(window.location.href).searchParams.get(QUERY_PARAM);
  if (key === null) return;
  try {
    if (key === "") {
      window.localStorage.removeItem(STORAGE_KEY);
    } else {
      window.localStorage.setItem(STORAGE_KEY, key);
    }
  } catch {
    // storage unavailable — requests just won't carry the key
  }
}

/** Remove ?internal from the URL. Call after hydration — during render the
 * Next router still re-syncs the URL from its own state, undoing the strip. */
export function stripInternalKeyFromUrl(): void {
  const url = new URL(window.location.href);
  if (!url.searchParams.has(QUERY_PARAM)) return;
  url.searchParams.delete(QUERY_PARAM);
  window.history.replaceState(window.history.state, "", url.toString());
}
