// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * Color utility functions for benchmark visualizations
 */

import { modelColors, providerColors } from '@/lib/config/colors';

/**
 * Fallback colors for models/providers not in the predefined color maps
 */
const fallbackModelColors = [
  "#E74C3C",
  "#F39C12",
  "#3498DB",
  "#2ECC71",
  "#9B59B6",
  "#E67E22",
  "#1ABC9C",
  "#95A5A6",
  "#34495E",
  "#E91E63",
  "#FF9800",
  "#4CAF50",
  "#673AB7",
  "#FF5722",
  "#607D8B",
  "#795548",
  "#009688",
  "#FFC107"
];

/**
 * Fallback colors for providers not in the predefined color map
 */
const fallbackProviderColors = [
  "#8BC34A",
  "#CDDC39",
  "#FFEB3B",
  "#FF4081",
  "#7C4DFF",
  "#448AFF",
  "#40E0D0",
  "#DA70D6",
  "#98FB98",
  "#F0E68C",
  "#DDA0DD",
  "#87CEEB"
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
 * Get color for a model, using predefined colors or generating a consistent fallback
 * @param model - Model name
 * @returns Hex color code
 */
export function getModelColor(model: string): string {
  if (modelColors[model]) {
    return modelColors[model];
  }

  const hash = hashCode(model);
  return fallbackModelColors[Math.abs(hash) % fallbackModelColors.length] ?? "#E74C3C";
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
  return fallbackProviderColors[Math.abs(hash) % fallbackProviderColors.length] ?? "#8BC34A";
}
