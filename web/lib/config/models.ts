// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export const modelDisplayNames: Record<string, string> = {
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
  "ink-2": "Ink 2",
  default: "Default",
  enhanced: "Enhanced"
};

export const sttProviderNames: Record<string, string> = {
  assemblyai: "AssemblyAI",
  cartesia: "Cartesia",
  deepgram: "Deepgram",
  elevenlabs: "ElevenLabs",
  gradium: "Gradium",
  openai: "OpenAI",
  speechmatics: "Speechmatics"
};

export const ttsProviderNames: Record<string, string> = {
  cartesia: "Cartesia",
  deepgram: "Deepgram",
  elevenlabs: "ElevenLabs",
  gradium: "Gradium",
  hume: "Hume",
  openai: "OpenAI",
  rime: "Rime",
  xai: "xAI"
};
