// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export const modelDisplayNames: Record<string, string> = {
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
  scribe_v2_realtime: "Scribe v2",
  "universal-streaming": "Universal Streaming",
  default: "Default",
  enhanced: "Enhanced"
};

export const sttProviderNames: Record<string, string> = {
  assemblyai: "AssemblyAI",
  deepgram: "Deepgram",
  elevenlabs: "ElevenLabs",
  speechmatics: "Speechmatics"
};
