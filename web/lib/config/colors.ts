// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

// Series colors are a soft, pastel-leaning categorical palette drawn in the
// spirit of the Coval brand families (blue / green / red-orange / purple),
// extended with on-vibe intermediate hues (teal, amber, indigo, rose, sky,
// lime) so the ~16 vendors stay distinguishable. Values are placed in OKLCH at
// the softest lightness that still clears the dataviz validator's categorical
// band (L 0.43–0.77, chroma ≥ 0.10) so they read as gentle pastels while
// surviving as thin line strokes on the near-white surface. Each vendor owns a
// hue family; its models step through shades of that family. Where a modality
// packs more vendors than hues, two vendors share a family but sit in opposite
// shade ranges (legend, tooltips, and export labels disambiguate). Generated
// from OKLCH anchors — see the brand guide color families.

export const modelColors: Record<string, string> = {
  // OpenAI — rose. TTS, STT, and S2S span the rose range. Cross-hosted models
  // follow their lab's family, so Together-hosted Whisper sits here too.
  "gpt-4o-mini-tts": "#de6d9b",
  "gpt-4o-transcribe": "#f782b1",
  "gpt-4o-mini-transcribe": "#d1608f",
  "gpt-realtime-whisper": "#b24574",
  "gpt-realtime": "#c35483", // S2S
  "together:whisper-large-v3": "#8f3055",

  // Google — sky (light blue). Chirp STT + Gemini realtime.
  chirp_2: "#53b0de",
  chirp_3: "#1d85b0",

  // ElevenLabs — peach (red-orange). TTS + STT scribe span light→dark peach.
  eleven_multilingual_v2: "#fb9167",
  eleven_flash_v2_5: "#d87248",
  eleven_turbo_v2_5: "#bd592f",
  eleven_v3: "#b04d20",
  scribe_v2_realtime: "#e67e54",

  // xAI — purple.
  "grok-tts": "#c890ea",
  "grok-stt": "#844da2",

  // Cartesia — blue (TTS sonic + STT ink).
  "sonic-3.5": "#5e93e1",
  "ink-2": "#3b6eb9",

  // Rime — green, dark range (separates from Gradium in TTS).
  arcana: "#55a14e",
  coda: "#3c8836",
  mistv3: "#22701c",

  // Hume — sky (TTS). Sits alongside Google, which is STT-only, so they never
  // co-occur in one chart.
  "octave-2": "#59b7e4",
  "octave-tts": "#268bb6",

  // Inworld AI — indigo.
  "inworld-tts-1.5-mini": "#a9a5ff",
  "inworld-tts-1.5-max": "#8b85de",
  "inworld-tts-2": "#5b55a9",
  "inworld-stt-1": "#736dc3",

  // Deepgram — teal. Aura anchors TTS; STT models span light→dark teal.
  "aura-2-thalia-en": "#29b69e",
  "nova-2": "#49cdb4",
  "nova-3": "#1db098",
  "flux-general-en": "#039580",
  "flux-general-multi": "#007b69",

  // AssemblyAI — purple (STT), light range (separates from xAI's dark grok-stt).
  "universal-streaming-multilingual": "#d299f5",
  "universal-3.5-pro": "#bf86e0",
  "universal-3-pro": "#a871c9",
  "universal-streaming": "#965fb6",

  // Speechmatics — peach, dark range (STT; separates from ElevenLabs scribe).
  enhanced: "#c35f36",
  default: "#a24112",

  // Soniox — lime (green-gold).
  "tts-rt-v1": "#9db030",
  "stt-rt-v4": "#859704",
  "stt-rt-v5": "#697800",

  // Smallest — amber.
  "lightning_v3.1_pro": "#d7a03d",
  pulse: "#b07b05",

  // Gladia — blue, mid (STT; sits between Cartesia's light sonic and dark ink).
  "solaria-1": "#6a9fee",

  // Mistral — amber, dark (STT; separates from Smallest's pulse).
  "voxtral-mini-transcribe-realtime-2602": "#835a01",

  // Azure — blue, dark (STT; separates from Cartesia's sonic/ink and Gladia).
  "azure:default": "#3d7fd1",

  "qwen3-tts-flash-realtime": "#9c6c03",

  // Together AI — green, dark range (STT; Rime's greens are TTS-only and
  // Gradium's STT default sits in the light range, so they stay apart).
  "nemotron-3-asr-streaming-0.6b": "#62ab5b",
  "nemotron-3.5-asr-streaming-0.6b": "#4a9243",
  "parakeet-tdt-0.6b-v3": "#33792c",

  // Composite keys for models whose slug is shared across providers
  "speechmatics:default": "#a24112",
  "gradium:default": "#74c16c"
};

export const providerColors: Record<string, string> = {
  OpenAI: "#d1608f",
  ElevenLabs: "#e67e54",
  Cartesia: "#5e93e1",
  Deepgram: "#1db098",
  AssemblyAI: "#bf86e0",
  Speechmatics: "#c35f36",
  Rime: "#3c8836",
  Gradium: "#74c16c",
  Hume: "#59b7e4",
  "Fish Audio": "#ec79a8",
  MiniMax: "#3cc3ab",
  "Inworld AI": "#8b85de",
  xAI: "#c890ea",
  Soniox: "#859704",
  Google: "#53b0de",
  Gladia: "#6a9fee",
  Mistral: "#835a01",
  Smallest: "#d7a03d",
  Azure: "#3d7fd1",
  Alibaba: "#9c6c03",
  "Together AI": "#4a9243"
};
