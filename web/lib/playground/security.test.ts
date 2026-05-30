// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";

function clearRedisEnv() {
  for (const key of Object.keys(process.env)) {
    if (
      key === "UPSTASH_REDIS_REST_URL" ||
      key === "UPSTASH_REDIS_REST_TOKEN" ||
      key === "KV_REST_API_URL" ||
      key === "KV_REST_API_TOKEN" ||
      key.endsWith("KV_REST_API_URL") ||
      key.endsWith("KV_REST_API_TOKEN")
    ) {
      delete process.env[key];
    }
  }
}

async function loadSecurityFallback() {
  vi.resetModules();
  clearRedisEnv();
  return import("./security");
}

beforeEach(() => {
  vi.resetModules();
});

describe("security (in-process fallback)", () => {
  it("allows up to MAX_CONCURRENT_PER_SESSION then rejects", async () => {
    const { MAX_CONCURRENT_PER_SESSION, tryAcquireSession } = await loadSecurityFallback();
    const sid = "s";
    for (let i = 0; i < MAX_CONCURRENT_PER_SESSION; i++) {
      expect(await tryAcquireSession(sid)).toBe(true);
    }
    expect(await tryAcquireSession(sid)).toBe(false);
  });

  it("releaseSession frees a concurrency slot", async () => {
    const { MAX_CONCURRENT_PER_SESSION, tryAcquireSession, releaseSession } =
      await loadSecurityFallback();
    const sid = "s";
    for (let i = 0; i < MAX_CONCURRENT_PER_SESSION; i++) {
      await tryAcquireSession(sid);
    }
    expect(await tryAcquireSession(sid)).toBe(false);
    await releaseSession(sid);
    expect(await tryAcquireSession(sid)).toBe(true);
  });

  it("consumes daily quota up to the cap then rejects", async () => {
    const { STT_DAILY_CAP, tryConsumeDailyQuota } = await loadSecurityFallback();
    const sid = "s";
    expect(await tryConsumeDailyQuota(sid, "stt", STT_DAILY_CAP)).toBe(true);
    expect(await tryConsumeDailyQuota(sid, "stt", 1)).toBe(false);
  });

  it("rolls back an over-cap consume so later requests still fit", async () => {
    const { STT_DAILY_CAP, tryConsumeDailyQuota } = await loadSecurityFallback();
    const sid = "s";
    expect(await tryConsumeDailyQuota(sid, "stt", STT_DAILY_CAP - 1)).toBe(true);
    expect(await tryConsumeDailyQuota(sid, "stt", 2)).toBe(false);
    expect(await tryConsumeDailyQuota(sid, "stt", 1)).toBe(true);
  });

  it("treats a non-positive count as a no-op", async () => {
    const { tryConsumeDailyQuota } = await loadSecurityFallback();
    expect(await tryConsumeDailyQuota("s", "tts", 0)).toBe(true);
  });
});
