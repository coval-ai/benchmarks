// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { COOKIE_NAME, getSessionFromRequest, mintSession, verifySession } from "./session";

const TEST_SECRET = "dev-secret-32-bytes-base64-placeholder=";
const OTHER_SECRET = "different-secret-also-32-bytes-base64==";

beforeEach(() => {
  process.env.PLAYGROUND_SESSION_SECRET = TEST_SECRET;
  delete process.env.PLAYGROUND_SESSION_SECRET_PREVIOUS;
});

afterEach(() => {
  vi.useRealTimers();
});

describe("session", () => {
  it("mintSession produces a verifiable token", () => {
    const { token, sid } = mintSession();
    expect(verifySession(token)?.sid).toBe(sid);
  });

  it("verifySession rejects tampered payload", () => {
    const { token } = mintSession();
    const [body, sig] = token.split(".") as [string, string];
    const tamperedBody = body.slice(0, -1) + (body.endsWith("A") ? "B" : "A");
    expect(verifySession(`${tamperedBody}.${sig}`)).toBeNull();
  });

  it("verifySession rejects tampered signature", () => {
    const { token } = mintSession();
    const [body, sig] = token.split(".") as [string, string];
    const tamperedSig = sig.slice(0, -1) + (sig.endsWith("A") ? "B" : "A");
    expect(verifySession(`${body}.${tamperedSig}`)).toBeNull();
  });

  it("verifySession rejects expired tokens", () => {
    const { token } = mintSession();
    vi.useFakeTimers();
    vi.setSystemTime(new Date(Date.now() + 25 * 60 * 60 * 1000));
    expect(verifySession(token)).toBeNull();
  });

  it("verifySession rejects malformed tokens", () => {
    expect(verifySession(null)).toBeNull();
    expect(verifySession("")).toBeNull();
    expect(verifySession("not-a-token")).toBeNull();
    expect(verifySession("a.b.c")).toBeNull();
  });

  it("verifySession rejects tokens signed with a different secret", () => {
    const { token } = mintSession();
    process.env.PLAYGROUND_SESSION_SECRET = OTHER_SECRET;
    expect(verifySession(token)).toBeNull();
  });

  it("verifySession accepts tokens signed with PLAYGROUND_SESSION_SECRET_PREVIOUS during rotation", () => {
    const { token } = mintSession();
    process.env.PLAYGROUND_SESSION_SECRET = OTHER_SECRET;
    process.env.PLAYGROUND_SESSION_SECRET_PREVIOUS = TEST_SECRET;
    expect(verifySession(token)?.sid).toBeTruthy();
  });

  it("getSessionFromRequest reads the cookie", () => {
    const { token, sid } = mintSession();
    const req = new Request("https://x.test", {
      headers: { cookie: `other=1; ${COOKIE_NAME}=${token}; another=2` },
    });
    expect(getSessionFromRequest(req)?.sid).toBe(sid);
  });

  it("getSessionFromRequest returns null when cookie is absent", () => {
    expect(getSessionFromRequest(new Request("https://x.test"))).toBeNull();
    expect(
      getSessionFromRequest(new Request("https://x.test", { headers: { cookie: "other=1" } })),
    ).toBeNull();
  });

  it("mintSession throws if PLAYGROUND_SESSION_SECRET is unset", () => {
    delete process.env.PLAYGROUND_SESSION_SECRET;
    expect(() => mintSession()).toThrow(/PLAYGROUND_SESSION_SECRET/);
  });
});
