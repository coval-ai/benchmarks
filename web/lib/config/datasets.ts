// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

const DATASET_LABELS: Record<string, string> = {
  "stt-v1": "LibriSpeech",
  "stt-v2": "FLEURS",
  "stt-v3": "PipeCat (production)",
  "stt-wildasr-clean": "WildASR clean",
  "stt-wildasr-clipping": "WildASR clipping",
  "stt-wildasr-farfield": "WildASR far-field",
  "stt-wildasr-noisegap": "WildASR noise gaps",
  "stt-wildasr-phonecodec": "WildASR phone codec",
  "stt-wildasr-reverb": "WildASR reverb",
  "stt-wildasr-accent": "WildASR accents",
  // Matches AboutMethodology's dataset table.
  "tts-v1": "Text prompts",
};

// S2S aggregates are pinned to the multi-turn dataset everywhere (dashboard
// and overview) so legacy single-turn s2s-v1 rows never pool into V2V stats.
export const S2S_MULTITURN_DATASET = "s2s-multiturn-v1";

export function datasetLabel(id: string): string {
  return DATASET_LABELS[id] ?? id;
}

export function isPerturbationDataset(id: string): boolean {
  // WildASR clean is the undegraded baseline, so it groups with the full sets.
  return id.startsWith("stt-wildasr-") && id !== "stt-wildasr-clean";
}
