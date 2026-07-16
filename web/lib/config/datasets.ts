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
