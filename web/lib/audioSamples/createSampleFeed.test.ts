// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import { createSampleFeed, SampleFetchError } from "./createSampleFeed";
import {
  parseS2SManifest,
  s2sSampleFeed,
  visibleRecordings,
  type S2SSampleManifest,
} from "./s2sFeed";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function stubFetch(impl: () => Promise<Response>): void {
  vi.stubGlobal("fetch", vi.fn(impl));
}

const validManifest: S2SSampleManifest = {
  bucket_at: "2026-07-19T00:00:00Z",
  test_case_id: "tc1",
  transcript: "hello",
  input_audio_url: null,
  recordings: [
    {
      provider: "openai",
      model: "gpt-realtime",
      object: "s2s-samples/x/openai.wav",
      coval_run_id: "r1",
      sim_id: "s1",
    },
  ],
};

const feed = createSampleFeed<S2SSampleManifest>(
  { name: "test", bucket: "b", prefix: "p", refetchMs: 1000 },
  parseS2SManifest
);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchIndex", () => {
  it("returns [] on 404 (nothing published yet)", async () => {
    stubFetch(() => Promise.resolve(new Response("", { status: 404 })));
    await expect(feed.fetchIndex()).resolves.toEqual([]);
  });

  it("sorts ticks newest-first regardless of source order", async () => {
    stubFetch(() =>
      Promise.resolve(
        jsonResponse([
          "2026-07-01T00:00:00Z",
          "2026-07-03T00:00:00Z",
          "2026-07-02T00:00:00Z",
        ])
      )
    );
    await expect(feed.fetchIndex()).resolves.toEqual([
      "2026-07-03T00:00:00Z",
      "2026-07-02T00:00:00Z",
      "2026-07-01T00:00:00Z",
    ]);
  });

  it("classifies a non-array body as a parse error", async () => {
    stubFetch(() => Promise.resolve(jsonResponse({ nope: true })));
    await expect(feed.fetchIndex()).rejects.toMatchObject({ kind: "parse" });
  });

  it("classifies an array with non-string entries as a parse error", async () => {
    stubFetch(() => Promise.resolve(jsonResponse(["a", 2])));
    await expect(feed.fetchIndex()).rejects.toMatchObject({ kind: "parse" });
  });

  it("classifies a 5xx as an http error carrying the status", async () => {
    stubFetch(() => Promise.resolve(new Response("", { status: 503 })));
    await expect(feed.fetchIndex()).rejects.toMatchObject({ kind: "http", status: 503 });
  });

  it("classifies a rejected fetch as a network error", async () => {
    stubFetch(() => Promise.reject(new TypeError("Failed to fetch")));
    await expect(feed.fetchIndex()).rejects.toBeInstanceOf(SampleFetchError);
    stubFetch(() => Promise.reject(new TypeError("Failed to fetch")));
    await expect(feed.fetchIndex()).rejects.toMatchObject({ kind: "network" });
  });
});

describe("fetchManifest", () => {
  it("returns null on 404 (a genuine no-sample tick)", async () => {
    stubFetch(() => Promise.resolve(new Response("", { status: 404 })));
    await expect(feed.fetchManifest("t")).resolves.toBeNull();
  });

  it("returns the parsed manifest on 200", async () => {
    stubFetch(() => Promise.resolve(jsonResponse(validManifest)));
    await expect(feed.fetchManifest("t")).resolves.toEqual(validManifest);
  });

  it("classifies a 5xx as an http error, not a missing sample", async () => {
    stubFetch(() => Promise.resolve(new Response("", { status: 500 })));
    await expect(feed.fetchManifest("t")).rejects.toMatchObject({ kind: "http", status: 500 });
  });

  it("classifies malformed JSON as a parse error", async () => {
    stubFetch(() => Promise.resolve(new Response("{not json", { status: 200 })));
    await expect(feed.fetchManifest("t")).rejects.toMatchObject({ kind: "parse" });
  });

  it("rejects a structurally invalid manifest ({} would crash the UI)", async () => {
    stubFetch(() => Promise.resolve(jsonResponse({})));
    await expect(feed.fetchManifest("t")).rejects.toMatchObject({ kind: "parse" });
  });

  it("passes an aborted request through untouched", async () => {
    const abort = new DOMException("aborted", "AbortError");
    stubFetch(() => Promise.reject(abort));
    await expect(feed.fetchManifest("t")).rejects.toBe(abort);
  });
});

describe("parseS2SManifest", () => {
  it("accepts a valid manifest and null transcript/input", () => {
    const m = parseS2SManifest({ ...validManifest, transcript: null, input_audio_url: "u" });
    expect(m.transcript).toBeNull();
    expect(m.input_audio_url).toBe("u");
  });

  it("drops non-spoken turns from an already-published manifest", () => {
    const m = parseS2SManifest({
      ...validManifest,
      recordings: [
        {
          ...validManifest.recordings[0]!,
          turns: [
            { index: 0, role: "user", content: "hi" },
            { index: 1, role: "assistant", content: "hello" },
            { index: 2, role: "tool", content: '{"function": "end_conversation"}' },
          ],
        },
      ],
    });
    expect(m.recordings[0]!.turns?.map((t) => t.role)).toEqual(["user", "assistant"]);
  });

  it("reads turn offsets, and tolerates absent or null ones", () => {
    const m = parseS2SManifest({
      ...validManifest,
      recordings: [
        {
          ...validManifest.recordings[0]!,
          turns: [
            { index: 0, role: "user", content: "hi", start_offset: 1.5, end_offset: 2.25 },
            // Coval omitted the end; the runner writes null.
            { index: 1, role: "assistant", content: "hello", start_offset: 3, end_offset: null },
            // Published before offsets were captured — keys absent entirely.
            { index: 2, role: "user", content: "bye" },
          ],
        },
      ],
    });
    const turns = m.recordings[0]!.turns!;
    expect(turns.map((t) => t.start_offset)).toEqual([1.5, 3, undefined]);
    expect(turns.map((t) => t.end_offset)).toEqual([2.25, undefined, undefined]);
  });

  it.each([
    ["null", null],
    ["missing recordings", { bucket_at: "x", test_case_id: "y", transcript: null, input_audio_url: null }],
    [
      "a non-finite turn offset",
      {
        ...validManifest,
        recordings: [
          {
            ...validManifest.recordings[0]!,
            turns: [{ index: 0, role: "user", content: "hi", start_offset: "1.5" }],
          },
        ],
      },
    ],
    ["recordings not an array", { ...validManifest, recordings: {} }],
    ["recording missing a string field", { ...validManifest, recordings: [{ provider: "openai" }] }],
    ["transcript wrong type", { ...validManifest, transcript: 42 }],
  ])("throws on %s", (_label, bad) => {
    expect(() => parseS2SManifest(bad)).toThrow();
  });
});

describe("visibleRecordings", () => {
  it("keeps only recordings whose model is visible", () => {
    const manifest: S2SSampleManifest = {
      ...validManifest,
      recordings: [
        { ...validManifest.recordings[0]!, provider: "openai", model: "gpt-realtime" },
        { ...validManifest.recordings[0]!, provider: "hidden", model: "hidden-model" },
      ],
    };
    const out = visibleRecordings(manifest, new Set(["openai:gpt-realtime"]));
    expect(out).toHaveLength(1);
    expect(out[0]!.provider).toBe("openai");
  });

  it("distinguishes two models of the same provider", () => {
    const manifest: S2SSampleManifest = {
      ...validManifest,
      recordings: [
        { ...validManifest.recordings[0]!, provider: "xai", model: "grok-realtime" },
        { ...validManifest.recordings[0]!, provider: "xai", model: "grok-voice-think-fast-2.0" },
      ],
    };
    const out = visibleRecordings(manifest, new Set(["xai:grok-voice-think-fast-2.0"]));
    expect(out).toHaveLength(1);
    expect(out[0]!.model).toBe("grok-voice-think-fast-2.0");
  });
});

describe("SampleFetchError", () => {
  it("carries kind and status", () => {
    const err = new SampleFetchError("http", "boom", 403);
    expect(err).toBeInstanceOf(Error);
    expect(err.kind).toBe("http");
    expect(err.status).toBe(403);
  });
});

describe("s2sSampleFeed.objectUrl", () => {
  it("resolves a bucket-root-relative object path", () => {
    expect(s2sSampleFeed.objectUrl("s2s-samples/x/openai.wav")).toBe(
      "https://storage.googleapis.com/coval-benchmarks-s2s-samples/s2s-samples/x/openai.wav"
    );
  });
});
