// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { ProvidersApiResponse } from "../api/client";
import type { ModelsByProvider } from "../../types/benchmark.types";
import {
  buildFacetGroups,
  buildTagIndex,
  filterModelsByFacets,
  toggleFacetValue,
} from "./facets";

// Four STT models: every mode is realtime (single-value → not a facet); host
// and features vary, so those are the real facets. Cast because `tags` is read
// defensively in facets.ts and isn't part of the generated ModelInfo type.
function tag(category: string, value: string) {
  return { category, value };
}
const PROVIDERS = {
  stt: [
    {
      provider: "deepgram",
      models: [
        {
          model: "nova-2",
          tags: [tag("type", "STT"), tag("host", "deepgram"), tag("mode", "realtime"), tag("features", "multilingual"), tag("features", "vad")],
        },
        {
          model: "flux-general-en",
          tags: [tag("type", "STT"), tag("host", "deepgram"), tag("mode", "realtime"), tag("features", "vad")],
        },
      ],
    },
    {
      provider: "openai",
      models: [
        {
          model: "gpt-4o-transcribe",
          tags: [tag("type", "STT"), tag("host", "openai"), tag("mode", "realtime"), tag("features", "multilingual"), tag("features", "vad")],
        },
      ],
    },
    {
      provider: "cartesia",
      models: [
        {
          model: "ink-2",
          tags: [tag("type", "STT"), tag("host", "cartesia"), tag("mode", "realtime"), tag("features", "vad")],
        },
      ],
    },
  ],
  tts: [],
} as unknown as ProvidersApiResponse;

const ALL: ModelsByProvider = {
  deepgram: ["deepgram:nova-2", "deepgram:flux-general-en"],
  openai: ["openai:gpt-4o-transcribe"],
  cartesia: ["cartesia:ink-2"],
};

const index = () => buildTagIndex("STT", PROVIDERS);

describe("filterModelsByFacets", () => {
  it("returns everything when nothing is selected", () => {
    expect(filterModelsByFacets(ALL, index(), {})).toEqual(ALL);
  });

  it("ORs values within a category", () => {
    const out = filterModelsByFacets(ALL, index(), { host: ["deepgram", "openai"] });
    expect(out.deepgram).toHaveLength(2);
    expect(out.openai).toHaveLength(1);
    expect(out.cartesia).toBeUndefined();
  });

  it("ANDs across categories", () => {
    const out = filterModelsByFacets(ALL, index(), {
      host: ["deepgram", "openai"],
      features: ["multilingual"],
    });
    // flux-general-en is deepgram but not multilingual → dropped.
    expect(out.deepgram).toEqual(["deepgram:nova-2"]);
    expect(out.openai).toEqual(["openai:gpt-4o-transcribe"]);
  });
});

describe("buildFacetGroups", () => {
  const id = (s: string) => s;

  it("drops single-value categories (type, mode) and keeps host + features", () => {
    const groups = buildFacetGroups(ALL, index(), {}, id);
    expect(groups.map((g) => g.category)).toEqual(["host", "features"]);
  });

  it("counts honor the other categories' selection", () => {
    const groups = buildFacetGroups(ALL, index(), { features: ["multilingual"] }, id);
    const host = groups.find((g) => g.category === "host")!;
    const byValue = Object.fromEntries(host.options.map((o) => [o.value, o.count]));
    // Among multilingual models, deepgram has 1 (nova-2), openai 1, cartesia 0.
    expect(byValue).toEqual({ deepgram: 1, openai: 1, cartesia: 0 });
  });

  it("labels VAD specially and marks active options", () => {
    const groups = buildFacetGroups(ALL, index(), { features: ["vad"] }, id);
    const features = groups.find((g) => g.category === "features")!;
    const vad = features.options.find((o) => o.value === "vad")!;
    expect(vad.label).toBe("VAD");
    expect(vad.active).toBe(true);
  });
});

describe("toggleFacetValue", () => {
  it("adds, then removes and prunes the empty category", () => {
    const added = toggleFacetValue({}, "host", "openai");
    expect(added).toEqual({ host: ["openai"] });
    expect(toggleFacetValue(added, "host", "openai")).toEqual({});
  });
});
