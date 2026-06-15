// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export const modelColors: Record<string, string> = {
  // OpenAI — red family (#E74C3C). Both TTS and STT use the same red range.
  "gpt-4o-mini-tts": "#C0392B",
  "gpt-realtime-whisper": "#E74C3C",

  // ElevenLabs — orange family (#F39C12). TTS + STT scribe all span light→dark orange.
  eleven_multilingual_v2: "#F5B041",
  eleven_flash_v2_5: "#F39C12",
  eleven_turbo_v2_5: "#E67E22",
  scribe_v2_realtime: "#CA6F1E",

  // xAI — pink family (#EC4899).
  "grok-tts": "#F472B6",
  "grok-stt": "#BE185D",

  // Cartesia — blue family (#3498DB).
  "sonic-3": "#85C1E9",
  "sonic-3.5": "#3498DB",
  "ink-2": "#1A5276",

  // Rime — green family (#27AE60).
  arcana: "#82E0AA",
  coda: "#27AE60",
  mistv3: "#1A7A40",

  // Hume — violet family (#A855F7).
  "octave-2": "#C084FC",
  "octave-tts": "#7C3AED",

  // Inworld AI — indigo family (#6366F1).
  "inworld-tts-1.5-mini": "#818CF8",
  "inworld-tts-1.5-max": "#4F46E5",

  // Deepgram — teal family (#16A085). TTS aura anchors the dark end; STT models
  // span up to light teal. All shades are visibly teal (none near-black).
  "aura-2-thalia-en": "#0C6654",
  "nova-2": "#0E8A76",
  "flux-general-en": "#16A085",
  "flux-general-multi": "#1EC4A3",
  "nova-3": "#45D4B5",

  // AssemblyAI — amethyst family (#8E44AD).
  "universal-streaming": "#8E44AD",

  // Speechmatics — navy family (#21618C).
  enhanced: "#2E86C1",
  default: "#21618C",
  // Composite keys for models whose slug is shared across providers
  "speechmatics:default": "#21618C",
  "gradium:default": "#1ABC9C"
};

export const providerColors: Record<string, string> = {
  OpenAI: "#E74C3C",
  ElevenLabs: "#F39C12",
  Cartesia: "#3498DB",
  Deepgram: "#16A085",
  AssemblyAI: "#8E44AD",
  Speechmatics: "#21618C",
  Rime: "#27AE60",
  Gradium: "#1ABC9C",
  Hume: "#A855F7",
  "Inworld AI": "#6366F1",
  xAI: "#EC4899"
};
