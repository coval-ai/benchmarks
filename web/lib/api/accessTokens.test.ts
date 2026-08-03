// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";

import { applyTokensFromUrl, captureTokensFromUrl, tokenHeaders } from "./accessTokens";

// The suite runs without a DOM, so stub only what the module touches. Storage
// persists across setUrl calls, the way a real browser's does.
const store = new Map<string, string>();

function setUrl(url: string): void {
  const location = { href: url };
  (globalThis as { window?: unknown }).window = {
    location,
    localStorage: {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
    },
    // replaceState rewrites the address bar without navigating, so the stub just
    // moves href — that is what the strip is observed through.
    history: { state: null, replaceState: (_s: unknown, _t: string, next: string) => {
      location.href = next;
    } },
  };
}

beforeEach(() => {
  store.clear();
  setUrl("https://benchmarks.coval.ai/s2s");
});

describe("captureTokensFromUrl", () => {
  it("stores a token from the URL and reports the change", () => {
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-token");
    expect(captureTokensFromUrl()).toBe(true);
    expect(tokenHeaders()).toEqual({ "X-EA-Token": "partner-token" });
  });

  it("reports no change when the same token is seen again", () => {
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-token");
    expect(captureTokensFromUrl()).toBe(true);
    expect(captureTokensFromUrl()).toBe(false);
  });

  it("reports a change when the token is swapped", () => {
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-one");
    captureTokensFromUrl();
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-two");
    expect(captureTokensFromUrl()).toBe(true);
    expect(tokenHeaders()).toEqual({ "X-EA-Token": "partner-two" });
  });

  it("treats an empty value as a clear, and reports it", () => {
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-token");
    captureTokensFromUrl();
    setUrl("https://benchmarks.coval.ai/s2s?ea=");
    expect(captureTokensFromUrl()).toBe(true);
    expect(tokenHeaders()).toEqual({});
  });

  it("reports no change when clearing an already-absent token", () => {
    setUrl("https://benchmarks.coval.ai/s2s?ea=");
    expect(captureTokensFromUrl()).toBe(false);
  });

  it("reports no change when no token param is present", () => {
    expect(captureTokensFromUrl()).toBe(false);
    expect(tokenHeaders()).toEqual({});
  });

  it("keeps the two tokens independent", () => {
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-token&internal=team-key");
    expect(captureTokensFromUrl()).toBe(true);
    expect(tokenHeaders()).toEqual({
      "X-Internal-Key": "team-key",
      "X-EA-Token": "partner-token",
    });

    setUrl("https://benchmarks.coval.ai/s2s?ea=");
    expect(captureTokensFromUrl()).toBe(true);
    expect(tokenHeaders()).toEqual({ "X-Internal-Key": "team-key" });
  });
});

describe("applyTokensFromUrl", () => {
  it("signals the identity change and strips the param", () => {
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-token");
    const onChange = vi.fn();
    applyTokensFromUrl(onChange);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(tokenHeaders()).toEqual({ "X-EA-Token": "partner-token" });
    expect(window.location.href).toBe("https://benchmarks.coval.ai/s2s");
  });

  it("still signals when Strict Mode replays the effect", () => {
    // Strict Mode mounts, unmounts and remounts, so the effect body runs twice.
    // The second pass sees the token already stored: the signal must survive the
    // first pass rather than being recomputed, which is why capture cannot live in
    // render — there, React keeps the second pass and the change is lost.
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-token");
    const onChange = vi.fn();
    applyTokensFromUrl(onChange);
    applyTokensFromUrl(onChange);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(tokenHeaders()).toEqual({ "X-EA-Token": "partner-token" });
  });

  it("signals again when a later token differs", () => {
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-one");
    const onChange = vi.fn();
    applyTokensFromUrl(onChange);
    setUrl("https://benchmarks.coval.ai/s2s?ea=partner-two");
    applyTokensFromUrl(onChange);
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(tokenHeaders()).toEqual({ "X-EA-Token": "partner-two" });
  });

  it("does not signal when there is no token to adopt", () => {
    const onChange = vi.fn();
    applyTokensFromUrl(onChange);
    expect(onChange).not.toHaveBeenCalled();
  });
});
