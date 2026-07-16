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
};

export function datasetLabel(id: string): string {
  return DATASET_LABELS[id] ?? id;
}

export function isPerturbationDataset(id: string): boolean {
  // WildASR clean is the undegraded baseline, so it groups with the full sets.
  return id.startsWith("stt-wildasr-") && id !== "stt-wildasr-clean";
}

export type WerBarView = "cumulative" | "clean" | "production";

// cumulative pools every dataset; clean/production pin the bar chart to the
// clean-audio (WildASR clean) and conversational (PipeCat) sets respectively.
export const WER_BAR_VIEWS: { key: WerBarView; label: string; dataset: string | null }[] = [
  { key: "cumulative", label: "Cumulative", dataset: null },
  { key: "clean", label: "Clean", dataset: "stt-wildasr-clean" },
  { key: "production", label: "Production", dataset: "stt-v3" },
];
