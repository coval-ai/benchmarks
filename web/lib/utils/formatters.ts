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
 * Normalize model name for display
 * @param modelName - Raw model name from API
 * @returns Formatted model name for display
 */
export function normalizeModelName(modelName: string): string {
  // Mapping mirrors the backend's enabled provider matrix (runner/config.py).
  const modelMappings: Record<string, string> = {
    // TTS
    "tts-1-hd": "TTS 1 HD",
    eleven_multilingual_v2: "Multilingual v2",
    eleven_flash_v2_5: "Flash v2.5",
    eleven_turbo_v2_5: "Turbo v2.5",
    "sonic-3": "Sonic 3",
    "aura-2-thalia-en": "Aura 2",
    arcana: "Arcana",
    mistv3: "Mist v3",
    // STT
    "nova-2": "Nova 2",
    "nova-3": "Nova 3",
    "flux-general-en": "Flux",
    "flux-general-multi": "Flux Multilingual",
    scribe_v2_realtime: "Scribe v2",
    "universal-streaming": "Universal Streaming",
    default: "Default",
    enhanced: "Enhanced"
  };

  // Return mapped name if it exists
  if (modelMappings[modelName]) {
    return modelMappings[modelName];
  }

  // Fallback: automatic normalization for unmapped models
  return modelName
    .replace(/-/g, " ") // Replace hyphens with spaces
    .replace(/_/g, " ") // Replace underscores with spaces
    .replace(/\bv(\d+)/g, "v$1") // Keep version format (v2, v3)
    .replace(/\b\w+/g, (word) => {
      // Capitalize first letter of each word, except version numbers
      if (/^v\d+/.test(word)) return word;
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    });
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
    speechmatics: "Speechmatics"
  };

  const lower = providerName.toLowerCase();
  return mappings[lower] ?? providerName;
}
