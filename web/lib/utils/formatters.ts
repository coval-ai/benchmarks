// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * Formatting utility functions for benchmark data
 */

/**
 * Format a timestamp as HH:MM in the viewer's local timezone (24-hour).
 *
 * `Intl.DateTimeFormat` defaults to the runtime timezone when none is given,
 * so chart axes always reflect what time the viewer was looking at the page —
 * not UTC, not the server's TZ.
 */
export function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });
}

/** Short month-day label (e.g. "Jun 5") for date-scale axis ticks. */
export function formatDate(timestamp: number): string {
  return new Date(timestamp).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric"
  });
}

/**
 * Same as {@link formatTime} but includes seconds — used in tooltips where
 * the user is hovering a specific data point and wants higher precision.
 */
export function formatTimeWithSeconds(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });
}

/**
 * Short timezone abbreviation for the viewer's locale (e.g. "PDT", "UTC", "GMT+1").
 * Returned as a plain string so callers can interpolate into axis labels.
 */
export function getLocalTimeZoneAbbr(): string {
  try {
    const parts = new Intl.DateTimeFormat(undefined, {
      timeZoneName: "short"
    }).formatToParts(new Date());
    return parts.find((p) => p.type === "timeZoneName")?.value ?? "";
  } catch {
    return "";
  }
}

/**
 * Convert a latency metric value to display milliseconds.
 * STT latency metrics (TTFT) are stored in seconds; TTS (TTFA) already in ms.
 */
export function latencyToMs(value: number, benchmark: "tts" | "stt"): number {
  return benchmark === "stt" ? value * 1000 : value;
}

/**
 * Build a composite model key from provider and model slug.
 * Used throughout the frontend to uniquely identify a (provider, model) pair,
 * since multiple providers can share the same model slug (e.g. "default").
 */
export function toModelKey(provider: string, model: string): string {
  return `${provider}:${model}`;
}

/**
 * Parse a composite model key back into its provider and model slug parts.
 * Returns the key unchanged as `model` when no colon separator is found.
 */
export function parseModelKey(key: string): { provider: string; model: string } {
  const idx = key.indexOf(":");
  if (idx === -1) return { provider: "", model: key };
  return { provider: key.slice(0, idx), model: key.slice(idx + 1) };
}

/**
 * Normalize model name for display.
 * Accepts both bare slugs ("default") and composite keys ("speechmatics:default").
 * @param modelKey - Model slug or composite "provider:model" key
 * @returns Formatted model name for display
 */
export function normalizeModelName(modelKey: string): string {
  const { model } = parseModelKey(modelKey);
  // Display labels only — sidebar/chart membership comes from result rows, not this map.
  const modelMappings: Record<string, string> = {
    // TTS
    "gpt-4o-mini-tts": "GPT-4o mini TTS",
    eleven_multilingual_v2: "Multilingual v2",
    eleven_flash_v2_5: "Flash v2.5",
    eleven_turbo_v2_5: "Turbo v2.5",
    "sonic-3": "Sonic 3",
    "sonic-3.5": "Sonic 3.5",
    "aura-2-thalia-en": "Aura 2",
    arcana: "Arcana",
    mistv3: "Mist v3",
    "octave-tts": "Octave TTS",
    "octave-2": "Octave 2",
    coda: "Coda",
    "grok-tts": "Grok TTS",
    // STT
    "nova-2": "Nova 2",
    "nova-3": "Nova 3",
    "flux-general-en": "Flux",
    "flux-general-multi": "Flux Multilingual",
    scribe_v2_realtime: "Scribe v2 Realtime",
    "gpt-realtime-whisper": "GPT Realtime Whisper",
    "universal-streaming": "Universal Streaming",
    default: "Default",
    enhanced: "Enhanced"
  };

  if (modelMappings[model]) {
    return modelMappings[model];
  }

  // Fallback: automatic normalization for unmapped models
  return model
    .replace(/-/g, " ") // Replace hyphens with spaces
    .replace(/_/g, " ") // Replace underscores with spaces
    .replace(/\bv(\d+)/g, "v$1") // Keep version format (v2, v3)
    .replace(/\b\w+/g, (word) => {
      // Capitalize first letter of each word, except version numbers
      if (/^v\d+/.test(word)) return word;
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    });
}

/** Display fallback for provider slugs not in the explicit maps below. */
function capitalizeProviderSlug(providerName: string): string {
  const trimmed = providerName.trim();
  if (!trimmed) return trimmed;
  return trimmed
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

/**
 * Normalize STT provider name for display
 * @param providerName - Raw provider name from API
 * @returns Formatted provider name for display
 */
export function normalizeSTTProviderName(providerName: string): string {
  const mappings: Record<string, string> = {
    assemblyai: "AssemblyAI",
    deepgram: "Deepgram",
    elevenlabs: "ElevenLabs",
    gradium: "Gradium",
    openai: "OpenAI",
    rime: "Rime",
    speechmatics: "Speechmatics"
  };

  const lower = providerName.toLowerCase();
  return mappings[lower] ?? capitalizeProviderSlug(providerName);
}

export function normalizeTTSProviderName(providerName: string): string {
  const mappings: Record<string, string> = {
    cartesia: "Cartesia",
    deepgram: "Deepgram",
    elevenlabs: "ElevenLabs",
    gradium: "Gradium",
    hume: "Hume",
    openai: "OpenAI",
    rime: "Rime",
    xai: "xAI",
  };

  const lower = providerName.toLowerCase();
  return mappings[lower] ?? capitalizeProviderSlug(providerName);
}
