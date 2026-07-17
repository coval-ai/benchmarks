// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * Color utility functions for benchmark visualizations
 */

import { modelColors, providerColors } from '../config/colors';
import { parseModelKey } from './formatters';

/**
 * Fallback colors for models/providers not in the predefined color maps.
 * The 10 validated pastel-leaning categorical anchors (blue, amber, purple,
 * teal, peach, indigo, green, rose, sky, lime), in a fixed order chosen for
 * adjacent colorblind separation. Used only when a model/provider isn't in the
 * predefined maps.
 */
const fallbackColors = [
  "#70a6f5",
  "#dea645",
  "#c289e4",
  "#3cc3ab",
  "#f78e64",
  "#9792ec",
  "#77c46f",
  "#ec79a8",
  "#59b7e4",
  "#a9bd40"
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

// Raw provider ids whose display name isn't a plain title-case of the slug.
// Everything else title-cases ("google" → "Google", "smallest" → "Smallest").
const PROVIDER_ID_ALIASES: Record<string, string> = {
  assemblyai: "AssemblyAI",
  elevenlabs: "ElevenLabs",
  openai: "OpenAI",
  xai: "xAI",
  inworld: "Inworld AI",
  fishaudio: "Fish Audio",
  minimax: "MiniMax",
  together: "Together AI",
};

/** Map a raw provider id (from a "provider:model" key) to its providerColors key. */
function providerDisplayFromId(id: string): string {
  const lower = id.toLowerCase();
  if (PROVIDER_ID_ALIASES[lower]) return PROVIDER_ID_ALIASES[lower];
  return lower
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

/**
 * Shift an sRGB hex toward white (amt > 0) or black (amt < 0) by |amt|, keeping
 * hue. Used to give a provider's unmapped models slightly different shades so
 * siblings stay distinguishable without leaving the family.
 */
function shiftLightness(hex: string, amt: number): string {
  const h = hex.replace("#", "");
  const target = amt >= 0 ? 255 : 0;
  const t = Math.abs(amt);
  const ch = (i: number) => {
    const c = parseInt(h.slice(i, i + 2), 16);
    return Math.round(c + (target - c) * t)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${ch(0)}${ch(2)}${ch(4)}`;
}

// Bare slugs that appear under more than one provider (via "provider:model"
// composite keys in modelColors), e.g. "default" for both Speechmatics and
// Gradium. For these the bare-slug entry belongs to one specific provider, so a
// composite key for a DIFFERENT provider must not borrow it — it should fall
// through to that provider's hue family instead.
const sharedSlugs = new Set(
  Object.keys(modelColors)
    .filter((key) => key.includes(":"))
    .map((key) => key.slice(key.indexOf(":") + 1))
);

/**
 * Get color for a model. Accepts both bare slugs ("default") and composite
 * "provider:model" keys ("speechmatics:default"). Composite keys are tried
 * first so that same-slug models across different providers can have distinct
 * colors.
 *
 * When a model isn't in the explicit map it inherits its PROVIDER's hue family
 * (with a small deterministic shade offset per model) rather than a random hash
 * color. This keeps a provider's models visually together — a newly added model
 * lands in the right family automatically, no per-model entry required. Only a
 * model whose provider is also unknown falls through to the neutral hash ramp.
 */
export function getModelColor(modelKey: string): string {
  if (modelColors[modelKey]) return modelColors[modelKey];
  const { provider, model } = parseModelKey(modelKey);
  // Skip the bare-slug entry for a composite key whose slug is shared across
  // providers — it belongs to another provider; use the hue family below.
  if (modelColors[model] && !(provider && sharedSlugs.has(model)))
    return modelColors[model];

  if (provider) {
    const base = providerColors[providerDisplayFromId(provider)];
    if (base) {
      // Deterministic per-model shade within the family: lighten/darken/keep.
      const step = ((Math.abs(hashCode(model)) % 3) - 1) * 0.16; // -0.16 | 0 | +0.16
      return step === 0 ? base : shiftLightness(base, step);
    }
  }

  if (provider) {
    // Unknown provider: hash the provider (not the model) into an anchor so
    // its models still share one family, shaded per model like above.
    const base =
      fallbackColors[Math.abs(hashCode(provider)) % fallbackColors.length] ?? "#70a6f5";
    const step = ((Math.abs(hashCode(model)) % 3) - 1) * 0.16;
    return step === 0 ? base : shiftLightness(base, step);
  }

  const hash = hashCode(modelKey);
  return fallbackColors[Math.abs(hash) % fallbackColors.length] ?? "#70a6f5";
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
  return fallbackColors[Math.abs(hash) % fallbackColors.length] ?? "#70a6f5";
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
