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
export function latencyToMs(value: number, benchmark: "tts" | "stt" | "s2s"): number {
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
    eleven_v3: "v3",
    "sonic-3": "Sonic 3",
    "sonic-3.5": "Sonic 3.5",
    "aura-2-thalia-en": "Aura 2",
    arcana: "Arcana",
    mistv3: "Mist v3",
    "octave-tts": "Octave TTS",
    "octave-2": "Octave 2",
    coda: "Coda",
    "grok-tts": "Grok TTS",
    "inworld-tts-2": "TTS 2",
    "inworld-tts-1.5-max": "TTS 1.5 Max",
    "inworld-tts-1.5-mini": "TTS 1.5 Mini",
    "tts-rt-v1": "TTS RT v1",
    neural: "Neural", // azure
    "dragon-hd-latest": "Dragon HD Latest", // azure
    // STT
    "nova-2": "Nova 2",
    "nova-3": "Nova 3",
    "flux-general-en": "Flux",
    "flux-general-multi": "Flux Multilingual",
    "grok-stt": "Grok STT",
    scribe_v2_realtime: "Scribe v2 Realtime",
    "gpt-realtime-whisper": "GPT Realtime Whisper",
    "gpt-4o-transcribe": "GPT-4o Transcribe",
    "gpt-4o-mini-transcribe": "GPT-4o mini Transcribe",
    "universal-streaming": "Universal Streaming",
    "universal-streaming-multilingual": "Universal Streaming Multilingual",
    "universal-3.5-pro": "Universal 3.5 Pro",
    "ink-2": "Ink 2",
    "stt-rt-v4": "STT RT v4",
    "stt-rt-v5": "STT RT v5",
    "inworld-stt-1": "STT 1",
    "nemotron-3-asr-streaming-0.6b": "Nemotron 3 ASR Streaming",
    "nemotron-3.5-asr-streaming-0.6b": "Nemotron 3.5 ASR Streaming",
    "parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3",
    "whisper-large-v3": "Whisper Large v3",
    // S2S
    "gpt-realtime": "GPT Realtime 2",
    "gemini-live": "Gemini 3.1 Flash Live (Preview)",
    "grok-realtime": "Grok Realtime",
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
      if (/^(tts|stt|s2s|hd|asr|api)$/i.test(word)) return word.toUpperCase();
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
    azure: "Azure",
    cartesia: "Cartesia",
    deepgram: "Deepgram",
    elevenlabs: "ElevenLabs",
    gradium: "Gradium",
    inworld: "Inworld AI",
    openai: "OpenAI",
    rime: "Rime",
    soniox: "Soniox",
    speechmatics: "Speechmatics",
    together: "Together AI",
    xai: "xAI"
  };

  const lower = providerName.toLowerCase();
  return mappings[lower] ?? capitalizeProviderSlug(providerName);
}

export function normalizeS2SProviderName(providerName: string): string {
  const mappings: Record<string, string> = {
    google: "Google",
    openai: "OpenAI",
    xai: "xAI",
  };

  const lower = providerName.toLowerCase();
  return mappings[lower] ?? capitalizeProviderSlug(providerName);
}

/** Pick the provider-name normalizer that matches the active dashboard tab. */
export function normalizeProviderNameForTab(
  providerName: string,
  tab: "tts" | "stt" | "s2s"
): string {
  if (tab === "stt") return normalizeSTTProviderName(providerName);
  if (tab === "s2s") return normalizeS2SProviderName(providerName);
  return normalizeTTSProviderName(providerName);
}

export function normalizeTTSProviderName(providerName: string): string {
  const mappings: Record<string, string> = {
    alibaba: "Alibaba",
    azure: "Azure",
    cartesia: "Cartesia",
    deepgram: "Deepgram",
    elevenlabs: "ElevenLabs",
    fishaudio: "Fish Audio",
    gradium: "Gradium",
    hume: "Hume",
    inworld: "Inworld AI",
    openai: "OpenAI",
    rime: "Rime",
    soniox: "Soniox",
    xai: "xAI",
  };

  const lower = providerName.toLowerCase();
  return mappings[lower] ?? capitalizeProviderSlug(providerName);
}
