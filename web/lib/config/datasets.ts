// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

const DATASET_LABELS: Record<string, string> = {
  "stt-v1": "LibriSpeech (read speech)",
  "stt-v2": "FLEURS (read speech)",
  "stt-v3": "PipeCat (conversational)",
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
  return id.startsWith("stt-wildasr-");
}

export type WerBarView = "cumulative" | "easy" | "hard";

// cumulative pools every dataset; easy/hard pin the bar chart to the clean-audio
// (WildASR clean) and conversational (PipeCat) sets respectively.
export const WER_BAR_VIEWS: { key: WerBarView; label: string; dataset: string | null }[] = [
  { key: "cumulative", label: "Cumulative", dataset: null },
  { key: "easy", label: "Easy", dataset: "stt-wildasr-clean" },
  { key: "hard", label: "Hard", dataset: "stt-v3" },
];
