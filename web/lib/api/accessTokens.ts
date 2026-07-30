// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * Access tokens that unlock early-access models.
 *
 * Landing on any page with `?internal=<key>` or `?ea=<token>` stores the value in
 * localStorage and strips it from the URL; every subsequent API request sends it as
 * a header. An empty value (`?ea=`) clears the stored one.
 *
 * `internal` is the benchmarking team's key and unlocks everything. `ea` is a
 * partner token: the server resolves it to an allowlist, so the browser sends an
 * opaque value and never names which models it wants to see.
 */

const TOKENS = {
  internal: { param: "internal", storage: "coval-internal-key", header: "X-Internal-Key" },
  ea: { param: "ea", storage: "coval-ea-token", header: "X-EA-Token" },
} as const;

type TokenKind = keyof typeof TOKENS;

function read(kind: TokenKind): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKENS[kind].storage);
  } catch {
    return null;
  }
}

/** Headers for every token currently stored. */
export function tokenHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const kind of Object.keys(TOKENS) as TokenKind[]) {
    const value = read(kind);
    if (value) headers[TOKENS[kind].header] = value;
  }
  return headers;
}

/** Store any token present in the URL. Safe to call during render (idempotent). */
export function captureTokensFromUrl(): void {
  const params = new URL(window.location.href).searchParams;
  for (const { param, storage } of Object.values(TOKENS)) {
    const value = params.get(param);
    if (value === null) continue;
    try {
      if (value === "") {
        window.localStorage.removeItem(storage);
      } else {
        window.localStorage.setItem(storage, value);
      }
    } catch {
      // storage unavailable — requests just won't carry the token
    }
  }
}

/** Remove token params from the URL. Call after hydration — during render the
 * Next router still re-syncs the URL from its own state, undoing the strip. */
export function stripTokensFromUrl(): void {
  const url = new URL(window.location.href);
  const present = Object.values(TOKENS).filter(({ param }) => url.searchParams.has(param));
  if (present.length === 0) return;
  for (const { param } of present) url.searchParams.delete(param);
  window.history.replaceState(window.history.state, "", url.toString());
}
