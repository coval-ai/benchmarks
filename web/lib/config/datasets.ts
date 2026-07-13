// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * Registry of the STT datasets the benchmark runs against. Each id is a
 * frozen, SHA-pinned manifest in the runner (see
 * runner/src/coval_bench/datasets/manifests/README.md). The API aggregates
 * per dataset; the dashboard isolates one at a time.
 */

export interface SttDataset {
  id: string;
  label: string;
  description: string;
}

export const STT_DATASETS: SttDataset[] = [
  {
    id: "stt-v1",
    label: "Easy",
    description:
      "LibriSpeech test-clean: studio-quality read speech most models train on.",
  },
  {
    id: "stt-v2",
    label: "Clean",
    description:
      "WildASR FLEURS clean: clean read speech, the long-running default set.",
  },
  {
    id: "stt-v3",
    label: "Hard",
    description:
      "Pipecat: spontaneous conversational speech from real voice-agent turns.",
  },
];

export const DEFAULT_STT_DATASET = "stt-v2";
