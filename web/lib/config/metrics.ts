// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export const metricDescriptions = {
  ttfa: {
    short: "Time to First Audio",
    detailed:
      "Delivering natural and responsive voice agents requires both speed and consistency. At Coval, we understand that latency is critical for realistic conversations, which is why we go beyond average measurements to track comprehensive percentile metrics with continuous 15-minute evaluation cycles. This rigorous approach ensures your voice AI maintains the reliable performance necessary for engaging user experiences."
  },
  ttft: {
    short: "Time to First Token",
    tooltip:
      "Time until the provider returns its first transcript token — how quickly partial results start streaming. Low TTFT with high TTFS points to slow finalization. Lower is better.",
    detailed:
      "We run TTFT measurements on a fixed test case to measure consistency over time. A model that consistently responds within a narrow time range provides better user experience than one with highly variable timing, even if the variable model is sometimes faster. Unpredictable delays can be seen through sudden latency spikes."
  },
  ttfs: {
    short: "Time to Final Segment",
    tooltip:
      "Time from end of speech until the final transcript is returned — the delay a voice agent actually waits before it can respond. Lower is better.",
    detailed:
      "TTFS measures how quickly a provider returns the final transcript once speech has ended, anchored at a shared VAD end-of-speech point so every provider is compared from the same instant. It isolates engine finalization speed — the latency a voice agent actually waits on before it can respond — independent of how long the speaker talked."
  },
  wer: {
    short: "Word Error Rate (%)",
    detailed:
      "Ensuring accurate speech output is fundamental to user trust and comprehension in voice AI systems. We recognize that even minor pronunciation errors can undermine the entire conversation experience and our evaluation captures how faithfully text-to-speech systems pronounce complex terminology, proper nouns, and domain-specific vocabulary that matter most to your users. Click a bar to highlight it for comparison."
  }
};
