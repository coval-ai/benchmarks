// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { ProvidersApiResponse } from "../api/client";
import type { Result } from "../aggregates";
import type { ModelsByProvider } from "../../types/benchmark.types";
import { toModelKey } from "./formatters";

function modelIsEnabled(
  providers: ProvidersApiResponse | undefined,
  benchmark: "STT" | "TTS",
  provider: string,
  model: string
): boolean {
  if (!providers) return true;

  const catalogue = benchmark === "STT" ? providers.stt : providers.tts;
  const providerInfo = catalogue.find((item) => item.provider === provider);
  if (!providerInfo) return true;

  const modelInfo = providerInfo.models.find((item) => item.model === model);
  return modelInfo?.disabled !== true;
}

function addModel(
  modelsByProvider: ModelsByProvider,
  provider: string,
  model: string
): void {
  const key = toModelKey(provider, model);
  const models = modelsByProvider[provider] ?? [];
  if (!models.includes(key)) {
    models.push(key);
    modelsByProvider[provider] = models;
  }
}

function buildModelsByProviderFromCatalogue(
  benchmark: "STT" | "TTS",
  providers?: ProvidersApiResponse
): ModelsByProvider {
  const modelsByProvider: ModelsByProvider = {};
  if (!providers) return modelsByProvider;

  const catalogue = benchmark === "STT" ? providers.stt : providers.tts;
  for (const providerInfo of catalogue) {
    for (const modelInfo of providerInfo.models) {
      if (modelInfo.disabled) continue;
      addModel(modelsByProvider, providerInfo.provider, modelInfo.model);
    }
  }

  return modelsByProvider;
}

export function buildModelsByProviderFromResults(
  rows: readonly Result[],
  benchmark: "STT" | "TTS",
  providers?: ProvidersApiResponse
): ModelsByProvider {
  const modelsByProvider = buildModelsByProviderFromCatalogue(benchmark, providers);

  for (const row of rows) {
    if (row.benchmark !== benchmark) continue;
    if (!modelIsEnabled(providers, benchmark, row.provider, row.model)) continue;
    addModel(modelsByProvider, row.provider, row.model);
  }

  return modelsByProvider;
}

/**
 * Drop any selected composite key that is no longer present in
 * `modelsByProvider`. Removal-only: it never adds or reorders, so it cannot
 * override a manual selection or re-introduce a model — it only reconciles the
 * selection after provider metadata filters a model out (e.g. a disabled model
 * that results alone had surfaced before the catalogue loaded).
 */
export function pruneSelection(
  selected: readonly string[],
  modelsByProvider: ModelsByProvider
): string[] {
  const valid = new Set(Object.values(modelsByProvider).flat());
  return selected.filter((key) => valid.has(key));
}
