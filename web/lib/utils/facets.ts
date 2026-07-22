// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { ModelTagOut, ProvidersApiResponse, TagCategoryOut } from "../api/client";
import type { ModelsByProvider } from "../../types/benchmark.types";
import { toModelKey } from "./formatters";
import { providerColors } from "../config/colors";
import { getModelColor } from "./colors";

/** category -> selected values. Within a category values OR; across categories they AND. */
export type FacetSelection = Record<string, string[]>;

/**
 * Reserved selection categories for individual models (the chart legend
 * toggles them). Not API tag categories, so they never render as sidebar chip
 * groups; a model matches them by its own composite key rather than a tag.
 * Picks show a model on top of the tag filter; excludes hide one from it.
 */
export const MODEL_FACET_CATEGORY = "model";
export const MODEL_EXCLUDE_CATEGORY = "model_hidden";

/**
 * Source facet identifiers, mirroring runner registries/tags.py. Shared by the
 * filter chips, the shared-graph exclusion below, and the dedicated-endpoint
 * chart styling.
 */
export const SOURCE_CATEGORY = "source";
export const DEDICATED_INFERENCE = "dedicated-inference";

export const DEDICATED_INFERENCE_LABEL = "Dedicated inference";
// Balanced by design: Baseten flagged fairness, so the shared side's
// advantages are stated alongside dedicated's. One string, every surface.
export const DEDICATED_INFERENCE_BLURB =
  "These endpoints run on capacity reserved for a single customer, so they appear on every chart except the shared latency timeline. Dedicated capacity typically delivers steadier response times, while shared endpoints serve many tenants, scale instantly, and reflect the out-of-the-box experience most users get.";

/** Whether a model's tags mark it as a dedicated-inference endpoint. */
export const isDedicated = (tags: ModelTagOut[]): boolean =>
  tags.some((t) => t.category === SOURCE_CATEGORY && t.value === DEDICATED_INFERENCE);

/** The composite keys of every dedicated-inference model in the catalogue. */
export function dedicatedModelKeys(tagIndex: Map<string, ModelTagOut[]>): Set<string> {
  const keys = new Set<string>();
  for (const [key, tags] of tagIndex) {
    if (isDedicated(tags)) keys.add(key);
  }
  return keys;
}

export interface FacetOption {
  value: string;
  label: string;
  count: number;
  maxCount: number;
  active: boolean;
  /** A legend-selected model carries this tag, so the chip rings without being toggled itself. */
  implied: boolean;
  color?: string;
}

export interface FacetGroup {
  category: string;
  label: string;
  options: FacetOption[];
}

/**
 * The facet vocabulary (categories, labels) — sourced from the API, except
 * creator is hoisted above host: who built a model matters more than who
 * serves it.
 */
export function getTagCategories(providers?: ProvidersApiResponse): TagCategoryOut[] {
  const categories = [...(providers?.tag_categories ?? [])];
  const creator = categories.findIndex((c) => c.category === "creator");
  const host = categories.findIndex((c) => c.category === "host");
  if (host !== -1 && creator > host) categories.splice(host, 0, ...categories.splice(creator, 1));
  return categories;
}

/** Map every catalogue model's composite key to its tags. */
export function buildTagIndex(
  benchmark: "STT" | "TTS" | "S2S",
  providers?: ProvidersApiResponse
): Map<string, ModelTagOut[]> {
  const index = new Map<string, ModelTagOut[]>();
  if (!providers) return index;
  const catalogue =
    (benchmark === "STT"
      ? providers.stt
      : benchmark === "S2S"
        ? providers.s2s
        : providers.tts) ?? [];
  for (const providerInfo of catalogue) {
    for (const modelInfo of providerInfo.models) {
      index.set(toModelKey(providerInfo.provider, modelInfo.model), modelInfo.tags ?? []);
    }
  }
  return index;
}

// A model passes when, for every category with a selection, it carries at least
// one of the selected values in that category (OR within, AND across).
function matchesSelection(tags: ModelTagOut[], selected: FacetSelection): boolean {
  for (const [category, values] of Object.entries(selected)) {
    if (values.length === 0) continue;
    const hit = tags.some((t) => t.category === category && values.includes(t.value));
    if (!hit) return false;
  }
  return true;
}

/** True when any category has at least one selected value. */
export const hasAnySelection = (selected: FacetSelection): boolean =>
  Object.values(selected).some((v) => v.length > 0);

/**
 * Keep only the models whose composite key is in `keys`. Used to restrict the
 * facet universe to models that actually have data to plot, so a chip can never
 * count (or filter to) a catalogue-only model that would show nothing.
 */
export function restrictToModelKeys(
  modelsByProvider: ModelsByProvider,
  keys: Set<string>
): ModelsByProvider {
  const out: ModelsByProvider = {};
  for (const [provider, modelKeys] of Object.entries(modelsByProvider)) {
    const kept = modelKeys.filter((key) => keys.has(key));
    if (kept.length > 0) out[provider] = kept;
  }
  return out;
}

/**
 * Narrow modelsByProvider to the models passing the current facet selection.
 * Legend model picks broaden an active tag filter (union) but stand alone
 * when no tag category is selected; excludes hide single models from either.
 * Dedicated-inference endpoints pass like any other model — only the latency
 * timeline excludes them, at the chart level.
 */
export function filterModelsByFacets(
  modelsByProvider: ModelsByProvider,
  tagIndex: Map<string, ModelTagOut[]>,
  selected: FacetSelection
): ModelsByProvider {
  if (!hasAnySelection(selected)) return modelsByProvider;
  const {
    [MODEL_FACET_CATEGORY]: picks = [],
    [MODEL_EXCLUDE_CATEGORY]: excludes = [],
    ...tagSelected
  } = selected;
  const hasTags = hasAnySelection(tagSelected);
  const out: ModelsByProvider = {};
  for (const [provider, keys] of Object.entries(modelsByProvider)) {
    const kept = keys.filter((key) => {
      if (excludes.includes(key)) return false;
      if (picks.includes(key)) return true;
      if (hasTags) return matchesSelection(tagIndex.get(key) ?? [], tagSelected);
      return picks.length === 0;
    });
    if (kept.length > 0) out[provider] = kept;
  }
  return out;
}

/**
 * Build the chip groups, in the API's category order. A category is shown only
 * when the visible models hold at least two distinct values for it. Each
 * option's count is how many models would remain if it were selected, honoring
 * the other categories' selection; labels and colors ignore the selection so
 * chips never rename or recolor while filtering.
 */
export function buildFacetGroups(
  modelsByProvider: ModelsByProvider,
  tagIndex: Map<string, ModelTagOut[]>,
  selected: FacetSelection,
  tagCategories: TagCategoryOut[],
  normalizeProvider: (name: string) => string
): FacetGroup[] {
  const visibleKeys = Object.values(modelsByProvider).flat();
  // Counts and active states follow only the tag categories; legend model
  // picks surface solely as the implied ring on provider chips.
  const modelSelection = selected[MODEL_FACET_CATEGORY] ?? [];
  const tagSelected = Object.fromEntries(
    Object.entries(selected).filter(
      ([c]) => c !== MODEL_FACET_CATEGORY && c !== MODEL_EXCLUDE_CATEGORY
    )
  );
  const groups: FacetGroup[] = [];

  for (const { category, label, provider_valued } of tagCategories) {
    const valueLabels = new Map<string, string>();
    const sharedValues = new Set<string>();
    for (const key of visibleKeys) {
      const tags = tagIndex.get(key) ?? [];
      for (const tag of tags) {
        if (tag.category !== category) continue;
        valueLabels.set(tag.value, tag.label);
        if (!isDedicated(tags)) sharedValues.add(tag.value);
      }
    }
    // A dedicated endpoint never mints a new filter group on its own: aside
    // from Source (the shared/dedicated axis itself), a category becomes a
    // facet only once shared endpoints hold two distinct values for it.
    if ((category === SOURCE_CATEGORY ? valueLabels.size : sharedValues.size) < 2)
      continue;

    const others: FacetSelection = { ...tagSelected, [category]: [] };
    const options: FacetOption[] = [...valueLabels.entries()]
      .map(([value, valueLabel]) => {
        const matching = visibleKeys.filter((key) =>
          (tagIndex.get(key) ?? []).some((t) => t.category === category && t.value === value)
        );
        const count = matching.filter((key) =>
          matchesSelection(tagIndex.get(key) ?? [], others)
        ).length;
        const label = provider_valued ? normalizeProvider(value) : valueLabel;
        let color: string | undefined;
        if (provider_valued) {
          // Chip follows the chart: derive the chip from the color of one of
          // its models so the sidebar can never drift from the series colors.
          color = (matching[0] ? getModelColor(matching[0]) : undefined) ?? providerColors[label];
        }
        return {
          value,
          label,
          count,
          maxCount: matching.length,
          active: (tagSelected[category] ?? []).includes(value),
          implied:
            !!provider_valued &&
            modelSelection.some((key) => matching.includes(key)),
          color,
        };
      })
      .sort((a, b) => a.label.localeCompare(b.label));

    groups.push({ category, label, options });
  }

  return groups;
}

export function toggleFacetValue(
  selected: FacetSelection,
  category: string,
  value: string
): FacetSelection {
  const current = selected[category] ?? [];
  const next = current.includes(value)
    ? current.filter((v) => v !== value)
    : [...current, value];
  const out = { ...selected };
  if (next.length > 0) out[category] = next;
  else delete out[category];
  return out;
}
