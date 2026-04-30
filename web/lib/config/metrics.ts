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
    detailed:
      "We run TTFT measurements on a fixed test case to measure consistency over time. A model that consistently responds within a narrow time range provides better user experience than one with highly variable timing, even if the variable model is sometimes faster. Unpredictable delays can be seen through sudden latency spikes."
  },
  latencyVariation: {
    short: "Latency distribution across all measurements",
    detailed:
      "This visualization shows the full distribution of latency measurements, including quartiles, outliers, and density curves. The violin shape reveals performance consistency - wider sections indicate more common latency values."
  },
  wer: {
    short: "Word Error Rate (%) \u2022 Click bar to compare models",
    detailed:
      "Ensuring accurate speech output is fundamental to user trust and comprehension in voice AI systems. We recognize that even minor pronunciation errors can undermine the entire conversation experience and our evaluation captures how faithfully text-to-speech systems pronounces complex terminology, proper nouns, and domain-specific vocabulary that matter most to your users."
  },
  rtf: {
    short: "Real-time factor",
    detailed:
      "RTF measures how fast a speech-to-text model processes audio relative to real-time. RTF = processing time / audio duration. Values < 1.0 mean faster than real-time, > 1.0 mean slower than real-time. Higher values are better for batch processing, while values close to 1.0 are ideal for real-time applications."
  },
  performanceGap: {
    short: "Time to First Token",
    detailed:
      "We measure how quickly the system returns the first transcribed token after the audio begins across multiple test cases. The fastest model is set as the baseline and we measure the delta relative to the fastest performer. Hover to view the first token."
  }
};
