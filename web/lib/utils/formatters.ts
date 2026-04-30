/**
 * Formatting utility functions for benchmark data
 */

/**
 * Format a timestamp to a readable time string
 * @param timestamp - Unix timestamp in milliseconds
 * @returns Formatted time string in HH:MM format (24-hour)
 */
export function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });
}

/**
 * Normalize model name for display
 * @param modelName - Raw model name from API
 * @returns Formatted model name for display
 */
export function normalizeModelName(modelName: string): string {
  // Define specific mappings for known models
  const modelMappings: Record<string, string> = {
    "tts-1": "TTS 1",
    "tts-1-hd": "TTS 1 HD",
    "gpt-4o-mini-tts": "GPT-4o Mini",
    "gpt-realtime-2025-08-28": "GPT-4o Realtime",
    eleven_multilingual_v2: "Multilingual v2",
    eleven_flash_v2_5: "Flash v2.5",
    eleven_turbo_v2_5: "Turbo v2.5",
    "sonic-2": "Sonic 2",
    "sonic-3": "Sonic 3",
    sonic: "Sonic",
    "sonic-turbo": "Sonic Turbo",
    mistv2: "Mist v2",
    mistv3: "Mist v3",
    arcana: "Arcana",
    "octave-tts": "Octave",
    "nova-2": "Nova 2",
    "nova-3": "Nova 3",
    universal: "Universal",
    "flux-general-en": "Flux",
    "aura-2-thalia-en": "Aura 2"
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
    speechmatics: "Speechmatics",
    google: "Google"
  };

  const lower = providerName.toLowerCase();
  return mappings[lower] ?? providerName;
}
