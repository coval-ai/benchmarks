// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

// Series colors are drawn from the Coval brand families (blue / green /
// red-orange / purple), expanded into legible mid-tones for the light surface
// and validated for colorblind separation. Each vendor owns a hue family; its
// models step through shades of that family. Where a modality packs more
// vendors than the four families allow, two vendors share a family but sit in
// opposite shade ranges (legend, tooltips, and export labels disambiguate).

export const modelColors: Record<string, string> = {
  // OpenAI — red family. TTS, STT, and S2S span the red range.
  "gpt-4o-mini-tts": "#93342a",
  "gpt-realtime-whisper": "#b23a2e",
  "gpt-4o-transcribe": "#cf6357",
  "gpt-4o-mini-transcribe": "#7e271f",
  "gpt-realtime": "#c04d40", // S2S

  // Google — blue family. Gemini realtime (S2S).
  "gemini-live": "#3f7fbf",

  // ElevenLabs — red-orange family. TTS + STT scribe span light→dark orange.
  eleven_multilingual_v2: "#d68a66",
  eleven_flash_v2_5: "#c15c3c",
  eleven_turbo_v2_5: "#a54c31",
  scribe_v2_realtime: "#8f3f26",

  // xAI — magenta (purple family).
  "grok-tts": "#cb7397",
  "grok-stt": "#7e3251",

  // Cartesia — blue family (TTS).
  "sonic-3": "#5b92cd",
  "sonic-3.5": "#2f6db0",
  "ink-2": "#1e4b7e",

  // Rime — green family.
  arcana: "#85ac56",
  coda: "#5e8a2e",
  mistv3: "#43631f",

  // Hume — purple family (TTS). Sits in the light/mid purple range so it stays
  // clear of Inworld's dark purples when both appear.
  "octave-2": "#9376c4",
  "octave-tts": "#6e51a6",

  // Inworld AI — purple family, dark range (separates from Hume in TTS and
  // AssemblyAI in STT).
  "inworld-tts-1.5-mini": "#5a417f",
  "inworld-tts-1.5-max": "#4b356f",
  "inworld-stt-1": "#3c2b5c",

  // Deepgram — green family, teal end. Aura anchors the dark end; STT models
  // span up to light teal.
  "aura-2-thalia-en": "#136452",
  "nova-2": "#17705c",
  "flux-general-en": "#1f937c",
  "flux-general-multi": "#2ca386",
  "nova-3": "#46b39a",

  // AssemblyAI — purple family (STT), light/mid range.
  "universal-streaming": "#8168b8",
  "universal-streaming-multilingual": "#9376c4",
  "universal-3.5-pro": "#6e51a6",

  // Speechmatics — blue family (STT).
  enhanced: "#3f7fbf",
  default: "#24598f",

  // Soniox — gold (red-orange family, yellow end). Reads distinctly warmer than
  // the ElevenLabs oranges and dark enough for the light surface.
  "tts-rt-v1": "#c0a03f",
  "stt-rt-v4": "#9e7b1c",
  "stt-rt-v5": "#6f5611",

  // Composite keys for models whose slug is shared across providers
  "speechmatics:default": "#24598f",
  "gradium:default": "#2ca386"
};

export const providerColors: Record<string, string> = {
  OpenAI: "#b23a2e",
  ElevenLabs: "#c15c3c",
  Cartesia: "#2f6db0",
  Deepgram: "#1f937c",
  AssemblyAI: "#6e51a6",
  Speechmatics: "#24598f",
  Rime: "#5e8a2e",
  Gradium: "#2ca386",
  Hume: "#8168b8",
  "Inworld AI": "#4b356f",
  xAI: "#b14a72",
  Soniox: "#9e7b1c",
  Google: "#3f7fbf"
};
