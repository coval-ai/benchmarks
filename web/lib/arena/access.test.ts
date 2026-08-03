// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { createHmac } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  type AccessPayload,
  LEGACY_LABELER_UNTIL,
  accessSecretConfigured,
  gateAllows,
  identified,
  legacyLabeler,
  mintAccess,
  needsRefresh,
  verifyAccess,
} from "./access";

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

  it("rejects an empty sid", () => {
    const forged = signed(encode({ sid: "", role: "labeler", iat: NOW, exp: NOW + 30 * DAY }));
    expect(verifyAccess(forged, NOW)).toBeNull();
  });

  it("rejects a payload carrying a role but no sid", () => {
    const forged = signed(encode({ role: "external", iat: NOW, exp: NOW + 30 * DAY }));
    expect(verifyAccess(forged, NOW)).toBeNull();
  });

  it("rejects a payload carrying a sid but no role", () => {
    const forged = signed(encode({ sid: "sid-1", iat: NOW, exp: NOW + 30 * DAY }));
    expect(verifyAccess(forged, NOW)).toBeNull();
  });

  it("rejects a token at the exact expiry instant", () => {
    expect(verifyAccess(mintAccess("external", "sid-1", NOW), NOW + 30 * DAY)).toBeNull();
  });
});

describe("arena gate", () => {
  it("lets a labeler reach the gated surfaces", () => {
    expect(gateAllows(verified(mintAccess("labeler", "sid-1", NOW)), NOW)).toBe(true);
  });

  it("keeps an external visitor out, signed cookie and all", () => {
    expect(gateAllows(verified(mintAccess("external", "sid-1", NOW)), NOW)).toBe(false);
  });

  it("honours a pre-identity cookie until the migration cutoff", () => {
    const legacy = signed(encode({ iat: NOW, exp: NOW + 30 * DAY }));
    expect(gateAllows(verified(legacy, NOW), NOW)).toBe(true);
  });

  // Verification rejects these outright, so this guards the second layer on its own: were
  // that check ever loosened, a half-filled payload still must not read as pre-identity.
  it("does not read a half-filled payload as a pre-identity cookie", () => {
    expect(legacyLabeler({ role: "external", iat: NOW, exp: NOW + 30 * DAY }, NOW)).toBe(false);
    expect(legacyLabeler({ sid: "sid-1", iat: NOW, exp: NOW + 30 * DAY }, NOW)).toBe(false);
  });

  it("stops honouring a pre-identity cookie once the cutoff passes", () => {
    const at = LEGACY_LABELER_UNTIL;
    const legacy = signed(encode({ iat: at, exp: at + 30 * DAY }));
    expect(gateAllows(verified(legacy, at), at)).toBe(false);
  });
});

describe("arena session secret", () => {
  it("treats a missing secret as unconfigured", () => {
    delete process.env.ARENA_SESSION_SECRET;
    expect(accessSecretConfigured()).toBe(false);
  });

  it("treats a secret shorter than 32 characters as unconfigured", () => {
    process.env.ARENA_SESSION_SECRET = "x".repeat(31);
    expect(accessSecretConfigured()).toBe(false);
  });

  it("accepts a secret of at least 32 characters", () => {
    process.env.ARENA_SESSION_SECRET = "x".repeat(32);
    expect(accessSecretConfigured()).toBe(true);
  });
});
