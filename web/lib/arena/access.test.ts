// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { createHmac } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type AccessPayload, identified, mintAccess, needsRefresh, verifyAccess } from "./access";

const DAY = 24 * 60 * 60 * 1000;
const NOW = 1_000_000;

beforeEach(() => {
  process.env.ARENA_SESSION_SECRET = "test-arena-session-secret";
});

afterEach(() => {
  delete process.env.ARENA_SESSION_SECRET_PREVIOUS;
});

/** Verify and assert, so tests read without non-null assertions everywhere. */
function verified(token: string, now: number = NOW): AccessPayload {
  const payload = verifyAccess(token, now);
  if (payload === null) throw new Error("expected a valid payload");
  return payload;
}

/** Sign an arbitrary payload body the way the module does, for hand-built cookies. */
function signed(body: string): string {
  const secret = process.env.ARENA_SESSION_SECRET ?? "";
  return `${body}.${createHmac("sha256", secret).update(body).digest("base64url")}`;
}

function encode(payload: Record<string, unknown>): string {
  return Buffer.from(JSON.stringify(payload)).toString("base64url");
}

describe("arena access cookie", () => {
  it("mints a token that verifies and carries a future expiry", () => {
    expect(verified(mintAccess("external", "sid-1", NOW), NOW + 1000).exp).toBeGreaterThan(NOW);
  });

  it("rejects a tampered signature", () => {
    const [body = ""] = mintAccess("external", "sid-1", NOW).split(".");
    expect(verifyAccess(`${body}.deadbeef`)).toBeNull();
  });

  it("rejects a tampered payload", () => {
    const [, sig = ""] = mintAccess("external", "sid-1", NOW).split(".");
    expect(verifyAccess(`${encode({ iat: 0, exp: 9e15 })}.${sig}`)).toBeNull();
  });

  it("rejects an expired token", () => {
    expect(verifyAccess(mintAccess("external", "sid-1", NOW), NOW + 31 * DAY)).toBeNull();
  });

  it("rejects a token signed with a different secret", () => {
    const token = mintAccess("external", "sid-1", NOW);
    process.env.ARENA_SESSION_SECRET = "a-different-secret";
    expect(verifyAccess(token, NOW + 1)).toBeNull();
  });

  it("returns null for missing or malformed tokens", () => {
    expect(verifyAccess(null)).toBeNull();
    expect(verifyAccess("not-a-dot-token")).toBeNull();
  });

  it("refreshes only after the refresh interval (1 day)", () => {
    const payload = verified(mintAccess("external", "sid-1", NOW), NOW);
    expect(needsRefresh(payload, NOW + 60_000)).toBe(false); // 1 min later
    expect(needsRefresh(payload, NOW + 2 * DAY)).toBe(true); // 2 days later
  });
});

describe("arena identity", () => {
  it("carries the sid and role it was minted with", () => {
    const payload = verified(mintAccess("labeler", "sid-42", NOW), NOW);
    expect(identified(payload)).toMatchObject({ sid: "sid-42", role: "labeler" });
  });

  it("keeps the same sid across a refresh so a voter stays recognisable", () => {
    const first = identified(verified(mintAccess("external", "sid-7", NOW), NOW));
    if (first === null) throw new Error("expected an identified payload");
    const rolled = verified(mintAccess("external", first.sid, NOW + 2 * DAY), NOW + 2 * DAY);
    expect(identified(rolled)?.sid).toBe("sid-7");
  });

  it("mints a distinct sid when none is supplied", () => {
    const now = Date.now();
    const a = identified(verified(mintAccess("external"), now))?.sid;
    const b = identified(verified(mintAccess("external"), now))?.sid;
    expect(a).not.toBe(b);
  });

  it("treats a pre-identity cookie as verified but unidentified", () => {
    const legacy = signed(encode({ iat: NOW, exp: NOW + 30 * DAY }));
    expect(identified(verified(legacy, NOW))).toBeNull();
  });

  it("rejects a role outside the known set", () => {
    const forged = signed(encode({ sid: "sid-1", role: "admin", iat: NOW, exp: NOW + 30 * DAY }));
    expect(verifyAccess(forged, NOW)).toBeNull();
  });

  it("accepts a cookie signed with the previous secret during rotation", () => {
    const old = mintAccess("labeler", "sid-9", NOW);
    process.env.ARENA_SESSION_SECRET_PREVIOUS = "test-arena-session-secret";
    process.env.ARENA_SESSION_SECRET = "rotated-secret";
    expect(identified(verified(old, NOW + 1))?.sid).toBe("sid-9");
  });
});
