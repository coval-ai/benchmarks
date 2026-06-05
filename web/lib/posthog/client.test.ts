// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";

const posthogMocks = vi.hoisted(() => ({
  capture: vi.fn(),
  getDistinctId: vi.fn(),
  getSessionId: vi.fn()
}));

vi.mock("posthog-js", () => ({
  default: {
    capture: posthogMocks.capture,
    get_distinct_id: posthogMocks.getDistinctId,
    get_session_id: posthogMocks.getSessionId
  }
}));

import { capturePostHogEvent } from "./client";

const originalToken = process.env.NEXT_PUBLIC_POSTHOG_TOKEN;

describe("posthog client helpers", () => {
  beforeEach(() => {
    posthogMocks.capture.mockReset();
    delete process.env.NEXT_PUBLIC_POSTHOG_TOKEN;
  });

  afterAll(() => {
    if (originalToken === undefined) {
      delete process.env.NEXT_PUBLIC_POSTHOG_TOKEN;
      return;
    }

    process.env.NEXT_PUBLIC_POSTHOG_TOKEN = originalToken;
  });

  it("skips capture when PostHog is not configured", () => {
    capturePostHogEvent("posthog_manual_test", { source: "test" });
    expect(posthogMocks.capture).not.toHaveBeenCalled();
  });

  it("captures events when PostHog is configured", () => {
    process.env.NEXT_PUBLIC_POSTHOG_TOKEN = "phc_test";

    capturePostHogEvent("posthog_manual_test", { source: "test" });

    expect(posthogMocks.capture).toHaveBeenCalledWith("posthog_manual_test", {
      source: "test"
    });
  });

  it("swallows capture errors so analytics never break the UI", () => {
    process.env.NEXT_PUBLIC_POSTHOG_TOKEN = "phc_test";
    posthogMocks.capture.mockImplementation(() => {
      throw new Error("capture failed");
    });

    expect(() => {
      capturePostHogEvent("posthog_manual_test", { source: "test" });
    }).not.toThrow();
  });
});
