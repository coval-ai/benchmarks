// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * Color utility functions for benchmark visualizations
 */

import { modelColors, providerColors } from '../config/colors';
import { parseModelKey } from './formatters';

/**
 * Fallback colors for models/providers not in the predefined color maps
 */
const fallbackModelColors = [
  "#2f6db0",
  "#1f937c",
  "#c15c3c",
  "#6e51a6",
  "#5e8a2e",
  "#b14a72",
  "#9e7b1c",
  "#b23a2e",
  "#5b92cd",
  "#46b39a",
  "#d68a66",
  "#9376c4",
  "#85ac56",
  "#cb7397",
  "#c0a03f",
  "#cf6357",
  "#1e4b7e",
  "#136452"
];

/**
 * Fallback colors for providers not in the predefined color map
 */
const fallbackProviderColors = [
  "#2f6db0",
  "#1f937c",
  "#c15c3c",
  "#6e51a6",
  "#5e8a2e",
  "#b14a72",
  "#9e7b1c",
  "#b23a2e",
  "#5b92cd",
  "#46b39a",
  "#d68a66",
  "#9376c4"
];

/**
 * Generate a hash code from a string
 * @param str - Input string
 * @returns Hash code as a number
 */
function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return hash;
}

/**
 * Get color for a model. Accepts both bare slugs ("default") and composite
 * "provider:model" keys ("speechmatics:default"). Composite keys are tried
 * first so that same-slug models across different providers can have distinct
 * colors; the hash fallback uses the full key for uniqueness.
 */
export function getModelColor(modelKey: string): string {
  if (modelColors[modelKey]) return modelColors[modelKey];
  const { model } = parseModelKey(modelKey);
  if (modelColors[model]) return modelColors[model];
  const hash = hashCode(modelKey);
  return fallbackModelColors[Math.abs(hash) % fallbackModelColors.length] ?? "#2f6db0";
}

/**
 * Get color for a provider, using predefined colors or generating a consistent fallback
 * @param provider - Provider name
 * @returns Hex color code
 */
export function getProviderColor(provider: string): string {
  if (providerColors[provider]) {
    return providerColors[provider];
  }

  const hash = hashCode(provider);
  return fallbackProviderColors[Math.abs(hash) % fallbackProviderColors.length] ?? "#2f6db0";
}

/**
 * Pick a light or dark foreground for a solid hex fill so text stays WCAG-AA
 * legible, using the relative-luminance crossover between black and white.
 */
export function getReadableTextColor(hex: string): string {
  const h = hex.replace("#", "");
  const channel = (i: number) => {
    const c = parseInt(h.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const luminance = 0.2126 * channel(0) + 0.7152 * channel(2) + 0.0722 * channel(4);
  return luminance > 0.179 ? "#0f0c0a" : "#ffffff";
}
