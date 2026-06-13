// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export const modelColors: Record<string, string> = {
  // OpenAI TTS — red family (#E74C3C)
  "gpt-4o-mini-tts": "#C0392B",

  // OpenAI STT — red family (#E74C3C)
  "gpt-realtime-whisper": "#F1948A",

  // ElevenLabs TTS — orange family (#F39C12)
  eleven_multilingual_v2: "#F5B041",
  eleven_flash_v2_5: "#F39C12",
  eleven_turbo_v2_5: "#CA6F1E",

  // ElevenLabs STT — orange family (#F39C12)
  scribe_v2_realtime: "#935116",

  // xAI TTS — pink family (#EC4899)
  "grok-tts": "#F472B6",

  // xAI STT — pink family (#EC4899)
  "grok-stt": "#BE185D",

  // Cartesia TTS — blue family (#3498DB)
  "sonic-3": "#85C1E9",
  "sonic-3.5": "#3498DB",

  // Cartesia STT — blue family (#3498DB)
  "ink-2": "#1A5276",

  // Rime TTS — green family (#27AE60)
  arcana: "#82E0AA",
  coda: "#27AE60",
  mistv3: "#1A7A40",

  // Hume TTS — violet family (#A855F7)
  "octave-2": "#C084FC",
  "octave-tts": "#7C3AED",

  // Deepgram TTS — teal family (#16A085)
  "aura-2-thalia-en": "#0A4F48",

  // Deepgram STT — teal family (#16A085)
  "nova-3": "#52C5AE",
  "nova-2": "#26AD94",
  "flux-general-multi": "#16A085",
  "flux-general-en": "#0D7065",

  // AssemblyAI STT — amethyst family (#8E44AD)
  "universal-streaming": "#8E44AD",

  // Speechmatics STT — navy family (#21618C)
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
  xAI: "#EC4899"
};
