// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { ModelTagOut, ProvidersApiResponse, TagCategoryOut } from "../api/client";
import type { ModelsByProvider } from "../../types/benchmark.types";
import { toModelKey } from "./formatters";
import { providerColors } from "../config/colors";
import { getModelColor } from "./colors";

/** category -> selected values. Within a category values OR; across categories they AND. */
export type FacetSelection = Record<string, string[]>;

export interface FacetOption {
  value: string;
  label: string;
  count: number;
  active: boolean;
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
  benchmark: "STT" | "TTS",
  providers?: ProvidersApiResponse
): Map<string, ModelTagOut[]> {
  const index = new Map<string, ModelTagOut[]>();
  if (!providers) return index;
  const catalogue = benchmark === "STT" ? providers.stt : providers.tts;
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

/** Narrow modelsByProvider to the models passing the current facet selection. */
export function filterModelsByFacets(
  modelsByProvider: ModelsByProvider,
  tagIndex: Map<string, ModelTagOut[]>,
  selected: FacetSelection
): ModelsByProvider {
  if (!hasAnySelection(selected)) return modelsByProvider;
  const out: ModelsByProvider = {};
  for (const [provider, keys] of Object.entries(modelsByProvider)) {
    const kept = keys.filter((key) => matchesSelection(tagIndex.get(key) ?? [], selected));
    if (kept.length > 0) out[provider] = kept;
  }
  return out;
}

/**
 * Build the chip groups, in the API's category order. A category is shown only
 * when the visible models hold at least two distinct values for it. Each
 * option's count is how many models would remain if it were selected, honoring
 * the other categories' selection.
 */
export function buildFacetGroups(
  modelsByProvider: ModelsByProvider,
  tagIndex: Map<string, ModelTagOut[]>,
  selected: FacetSelection,
  tagCategories: TagCategoryOut[],
  normalizeProvider: (name: string) => string
): FacetGroup[] {
  const visibleKeys = Object.values(modelsByProvider).flat();
  const groups: FacetGroup[] = [];

  for (const { category, label, provider_valued } of tagCategories) {
    const valueLabels = new Map<string, string>();
    for (const key of visibleKeys) {
      for (const tag of tagIndex.get(key) ?? []) {
        if (tag.category === category) valueLabels.set(tag.value, tag.label);
      }
    }
    if (valueLabels.size < 2) continue;

    const others: FacetSelection = { ...selected, [category]: [] };
    const options: FacetOption[] = [...valueLabels.entries()]
      .map(([value, valueLabel]) => {
        const count = visibleKeys.filter((key) => {
          const tags = tagIndex.get(key) ?? [];
          return (
            tags.some((t) => t.category === category && t.value === value) &&
            matchesSelection(tags, others)
          );
        }).length;
        const label = provider_valued ? normalizeProvider(value) : valueLabel;
        let color: string | undefined;
        if (provider_valued) {
          const repKey = visibleKeys.find((key) =>
            (tagIndex.get(key) ?? []).some((t) => t.category === category && t.value === value)
          );
          // Chip follows the chart: derive the host chip from the color of one
          // of its visible models so the sidebar can never drift from the
          // series colors. providerColors is only a fallback when no model of
          // the host is currently visible.
          color = (repKey ? getModelColor(repKey) : undefined) ?? providerColors[label];
        }
        return {
          value,
          label,
          count,
          active: (selected[category] ?? []).includes(value),
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
