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

/** The facet vocabulary (categories, labels, order) — sourced entirely from the API. */
export function getTagCategories(providers?: ProvidersApiResponse): TagCategoryOut[] {
  return providers?.tag_categories ?? [];
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
    for (const key of visibleKeys) {
      for (const tag of tagIndex.get(key) ?? []) {
        if (tag.category === category) valueLabels.set(tag.value, tag.label);
      }
    }
    if (valueLabels.size < 2) continue;

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
