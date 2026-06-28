// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { ProvidersApiResponse } from "../api/client";
import type { ModelsByProvider } from "../../types/benchmark.types";
import { toModelKey } from "./formatters";

// Defined locally, not from the generated schema, so the build never depends on
// the API having shipped `tags` yet — codegen rebuilds ModelInfo from the live
// API, and the frontend reads tags defensively (absent -> no facets).
export interface ModelTag {
  category: string;
  value: string;
}

/** category -> selected values. Within a category values OR; across categories they AND. */
export type FacetSelection = Record<string, string[]>;

export interface FacetOption {
  value: string;
  label: string;
  count: number;
  active: boolean;
}

export interface FacetGroup {
  category: string;
  label: string;
  options: FacetOption[];
}

// Display order and labels for the tag categories the API emits.
const CATEGORY_ORDER = ["type", "mode", "host", "lab", "features", "source", "tenancy"];
const CATEGORY_LABELS: Record<string, string> = {
  type: "Type",
  mode: "Mode",
  host: "Host",
  lab: "Lab",
  features: "Features",
  source: "Source",
  tenancy: "Tenancy",
};
// Values whose nice form isn't just a capitalization.
const VALUE_LABELS: Record<string, string> = { vad: "VAD" };

/** Map every catalogue model's composite key to its tags. */
export function buildTagIndex(
  benchmark: "STT" | "TTS",
  providers?: ProvidersApiResponse
): Map<string, ModelTag[]> {
  const index = new Map<string, ModelTag[]>();
  if (!providers) return index;
  const catalogue = benchmark === "STT" ? providers.stt : providers.tts;
  for (const providerInfo of catalogue) {
    for (const modelInfo of providerInfo.models) {
      const tags = (modelInfo as { tags?: ModelTag[] }).tags ?? [];
      index.set(toModelKey(providerInfo.provider, modelInfo.model), tags);
    }
  }
  return index;
}

function facetValueLabel(
  category: string,
  value: string,
  normalizeProvider: (name: string) => string
): string {
  if (category === "host" || category === "lab") return normalizeProvider(value);
  if (category === "type") return value.toUpperCase();
  return VALUE_LABELS[value] ?? value.charAt(0).toUpperCase() + value.slice(1);
}

// A model passes when, for every category with a selection, it carries at least
// one of the selected values in that category (OR within, AND across).
function matchesSelection(tags: ModelTag[], selected: FacetSelection): boolean {
  for (const [category, values] of Object.entries(selected)) {
    if (values.length === 0) continue;
    const hit = tags.some((t) => t.category === category && values.includes(t.value));
    if (!hit) return false;
  }
  return true;
}

const hasAnySelection = (selected: FacetSelection): boolean =>
  Object.values(selected).some((v) => v.length > 0);

/** Narrow modelsByProvider to the models passing the current facet selection. */
export function filterModelsByFacets(
  modelsByProvider: ModelsByProvider,
  tagIndex: Map<string, ModelTag[]>,
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
 * Build the chip groups. A category is shown only when it has at least two
 * distinct values across the visible models — a single-value facet can't filter
 * anything. Each option's count is the number of models that would remain if it
 * were selected, honoring the other categories' current selection.
 */
export function buildFacetGroups(
  modelsByProvider: ModelsByProvider,
  tagIndex: Map<string, ModelTag[]>,
  selected: FacetSelection,
  normalizeProvider: (name: string) => string
): FacetGroup[] {
  const visibleKeys = Object.values(modelsByProvider).flat();
  const groups: FacetGroup[] = [];

  for (const category of CATEGORY_ORDER) {
    const values = new Set<string>();
    for (const key of visibleKeys) {
      for (const tag of tagIndex.get(key) ?? []) {
        if (tag.category === category) values.add(tag.value);
      }
    }
    if (values.size < 2) continue;

    const others: FacetSelection = { ...selected, [category]: [] };
    const options: FacetOption[] = [...values]
      .map((value) => {
        const count = visibleKeys.filter((key) => {
          const tags = tagIndex.get(key) ?? [];
          return (
            tags.some((t) => t.category === category && t.value === value) &&
            matchesSelection(tags, others)
          );
        }).length;
        return {
          value,
          label: facetValueLabel(category, value, normalizeProvider),
          count,
          active: (selected[category] ?? []).includes(value),
        };
      })
      .sort((a, b) => a.label.localeCompare(b.label));

    groups.push({ category, label: CATEGORY_LABELS[category] ?? category, options });
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
