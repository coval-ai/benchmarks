// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { ProvidersApiResponse, TagCategoryOut } from "../api/client";
import type { ModelsByProvider } from "../../types/benchmark.types";
import {
  buildFacetGroups,
  buildTagIndex,
  filterModelsByFacets,
  getTagCategories,
  restrictToModelKeys,
  toggleFacetValue,
} from "./facets";

// Four STT models: every mode is realtime (single-value → not a facet); host
// and features vary, so those are the real facets. Each tag carries the label
// the API supplies; host is provider-valued so its label is the raw id.
function tag(category: string, value: string, label: string) {
  return { category, value, label };
}
const TYPE = tag("type", "STT", "STT");
const REALTIME = tag("mode", "realtime", "Realtime");
const MULTI = tag("features", "multilingual", "Multilingual");
const VAD = tag("features", "vad", "VAD");
const host = (id: string) => tag("host", id, id);

const TAG_CATEGORIES: TagCategoryOut[] = [
  { category: "type", label: "Type", provider_valued: false },
  { category: "mode", label: "Mode", provider_valued: false },
  { category: "host", label: "Host", provider_valued: true },
  { category: "lab", label: "Lab", provider_valued: true },
  { category: "features", label: "Features", provider_valued: false },
];

const PROVIDERS = {
  stt: [
    {
      provider: "deepgram",
      models: [
        { model: "nova-2", tags: [TYPE, host("deepgram"), REALTIME, MULTI, VAD] },
        { model: "flux-general-en", tags: [TYPE, host("deepgram"), REALTIME, VAD] },
      ],
    },
    {
      provider: "openai",
      models: [{ model: "gpt-4o-transcribe", tags: [TYPE, host("openai"), REALTIME, MULTI, VAD] }],
    },
    {
      provider: "cartesia",
      models: [{ model: "ink-2", tags: [TYPE, host("cartesia"), REALTIME, VAD] }],
    },
  ],
  tts: [],
  tag_categories: TAG_CATEGORIES,
} as unknown as ProvidersApiResponse;

const ALL: ModelsByProvider = {
  deepgram: ["deepgram:nova-2", "deepgram:flux-general-en"],
  openai: ["openai:gpt-4o-transcribe"],
  cartesia: ["cartesia:ink-2"],
};

const index = () => buildTagIndex("STT", PROVIDERS);
const cats = () => getTagCategories(PROVIDERS);

describe("getTagCategories", () => {
  it("hoists lab above host, keeping the rest in API order", () => {
    expect(cats().map((c) => c.category)).toEqual(["type", "mode", "lab", "host", "features"]);
  });
});

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

  it("drops single-value categories (type, mode) and keeps host + features in API order", () => {
    const groups = buildFacetGroups(ALL, index(), {}, cats(), id);
    expect(groups.map((g) => g.category)).toEqual(["host", "features"]);
  });

  it("counts honor the other categories' selection", () => {
    const groups = buildFacetGroups(ALL, index(), { features: ["multilingual"] }, cats(), id);
    const host = groups.find((g) => g.category === "host")!;
    const byValue = Object.fromEntries(host.options.map((o) => [o.value, o.count]));
    // Among multilingual models, deepgram has 1 (nova-2), openai 1, cartesia 0.
    expect(byValue).toEqual({ deepgram: 1, openai: 1, cartesia: 0 });
  });

  it("keeps labels, order, and colors stable while other categories filter", () => {
    const snapshot = (selected: Parameters<typeof buildFacetGroups>[2]) =>
      buildFacetGroups(ALL, index(), selected, cats(), id)
        .find((g) => g.category === "host")!
        .options.map(({ value, label, color, maxCount }) => ({ value, label, color, maxCount }));
    expect(snapshot({ features: ["multilingual"] })).toEqual(snapshot({}));
  });

  it("uses the API-supplied label and marks active options", () => {
    const groups = buildFacetGroups(ALL, index(), { features: ["vad"] }, cats(), id);
    const features = groups.find((g) => g.category === "features")!;
    const vad = features.options.find((o) => o.value === "vad")!;
    expect(vad.label).toBe("VAD");
    expect(vad.active).toBe(true);
  });

  it("renders provider-valued categories through normalizeProvider", () => {
    const groups = buildFacetGroups(ALL, index(), {}, cats(), (s) => s.toUpperCase());
    const host = groups.find((g) => g.category === "host")!;
    expect(host.options.map((o) => o.label)).toEqual(["CARTESIA", "DEEPGRAM", "OPENAI"]);
  });
});

describe("restrictToModelKeys", () => {
  it("drops models without data and prunes emptied providers", () => {
    // openai:gpt-4o-transcribe has no data → openai drops entirely; a chip
    // built over the result can't count a model that would chart nothing.
    const withData = new Set(["deepgram:nova-2", "cartesia:ink-2"]);
    expect(restrictToModelKeys(ALL, withData)).toEqual({
      deepgram: ["deepgram:nova-2"],
      cartesia: ["cartesia:ink-2"],
    });
  });

  it("hides a category once its only data-backed value collapses to one", () => {
    // Only deepgram models have data → host has a single value → not a facet.
    const withData = new Set(["deepgram:nova-2", "deepgram:flux-general-en"]);
    const universe = restrictToModelKeys(ALL, withData);
    const groups = buildFacetGroups(universe, index(), {}, cats(), (s) => s);
    expect(groups.map((g) => g.category)).not.toContain("host");
  });
});

describe("toggleFacetValue", () => {
  it("adds, then removes and prunes the empty category", () => {
    const added = toggleFacetValue({}, "host", "openai");
    expect(added).toEqual({ host: ["openai"] });
    expect(toggleFacetValue(added, "host", "openai")).toEqual({});
  });
});
