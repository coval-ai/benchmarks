// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export type MethodologyMetricKey = "ttfa" | "ttft" | "ttfs" | "wer";

export interface MethodologyChange {
  date: string;
  metrics?: MethodologyMetricKey[];
  title: string;
  detail: string;
}

export const methodologyChanges: MethodologyChange[] = [
  {
    date: "2026-06-10",
    metrics: ["ttfs"],
    title: "TTFS anchored at end-of-speech",
    detail:
      "Streaming STT finalization is now forced at a shared VAD end-of-speech point, so every provider is timed from the same instant. Steps after this date reflect engine finalization speed, not how long the speaker talked.",
  },
  {
    date: "2026-06-15",
    metrics: ["ttft"],
    title: "Grok excluded from TTFT",
    detail:
      "Grok has a ~1.1s fixed response floor that is not comparable to other providers' time-to-first-token. It is excluded from TTFT from this date onward.",
  },
  {
    date: "2026-06-15",
    metrics: ["ttfs"],
    title: "Gradium finalization made event-driven",
    detail:
      "Gradium STT now waits for an explicit finalization event instead of a fixed timeout, so its time-to-final-segment reflects true engine speed. Gradium TTFS values shift at this point.",
  },
];
