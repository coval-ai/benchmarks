// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export const modelColors: Record<string, string> = {
  // OpenAI TTS
  "tts-1-hd": "#943126",

  // ElevenLabs TTS
  eleven_multilingual_v2: "#FFD633",
  eleven_flash_v2_5: "#F39C12",
  eleven_turbo_v2_5: "#B7950B",

  // ElevenLabs STT
  scribe_v2_realtime: "#E67E22",

  // Cartesia TTS
  "sonic-3": "#3498DB",

  // Rime TTS
  arcana: "#33FF99",
  mistv3: "#0E5C32",

  // Deepgram TTS
  "aura-2-thalia-en": "#FC2D62",

  // Deepgram STT
  "nova-2": "#F39C12",
  "nova-3": "#FFD633",
  "flux-general-en": "#FFFF66",
  "flux-general-multi": "#FFB347",

  // AssemblyAI STT
  "universal-streaming": "#E74C3C",

  // Speechmatics STT
  enhanced: "#3498DB",
  default: "#21618C"
};

export const providerColors: Record<string, string> = {
  OpenAI: "#E74C3C",
  ElevenLabs: "#F39C12",
  Cartesia: "#3498DB",
  Deepgram: "#16A085",
  AssemblyAI: "#8E44AD",
  Speechmatics: "#21618C",
  Rime: "#27AE60",
  Gradium: "#1ABC9C"
};
