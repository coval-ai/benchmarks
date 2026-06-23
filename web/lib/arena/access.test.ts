// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it } from "vitest";
import { mintAccess, needsRefresh, verifyAccess } from "./access";

const DAY = 24 * 60 * 60 * 1000;

beforeEach(() => {
  process.env.ARENA_SESSION_SECRET = "test-arena-session-secret";
});

describe("arena access cookie", () => {
  it("mints a token that verifies and carries a future expiry", () => {
    const now = 1_000_000;
    const payload = verifyAccess(mintAccess(now), now + 1000);
    expect(payload).not.toBeNull();
    expect(payload?.exp).toBeGreaterThan(now);
  });

  it("rejects a tampered signature", () => {
    const [body = ""] = mintAccess(1_000_000).split(".");
    expect(verifyAccess(`${body}.deadbeef`)).toBeNull();
  });

  it("rejects a tampered payload", () => {
    const [, sig = ""] = mintAccess(1_000_000).split(".");
    const forged = Buffer.from(JSON.stringify({ iat: 0, exp: 9e15 })).toString("base64url");
    expect(verifyAccess(`${forged}.${sig}`)).toBeNull();
  });

  it("rejects an expired token", () => {
    const now = 1_000_000;
    expect(verifyAccess(mintAccess(now), now + 31 * DAY)).toBeNull();
  });

  it("rejects a token signed with a different secret", () => {
    const token = mintAccess(1_000_000);
    process.env.ARENA_SESSION_SECRET = "a-different-secret";
    expect(verifyAccess(token, 1_000_001)).toBeNull();
  });

  it("returns null for missing or malformed tokens", () => {
    expect(verifyAccess(null)).toBeNull();
    expect(verifyAccess("not-a-dot-token")).toBeNull();
  });

  it("refreshes only after the refresh interval (1 day)", () => {
    const now = 1_000_000;
    const payload = verifyAccess(mintAccess(now), now);
    if (payload === null) throw new Error("expected a valid payload");
    expect(needsRefresh(payload, now + 60_000)).toBe(false); // 1 min later
    expect(needsRefresh(payload, now + 2 * DAY)).toBe(true); // 2 days later
  });
});
