// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

// Definition of the active metric, shown as a bolded block inside the
// "About this benchmark" tooltip on cards that carry the TTFS/TTFT toggle —
// the toggle buttons themselves carry no tooltips. Metrics without a tooltip
// (TTFA, V2V) resolve to undefined and add nothing.
export const metricAboutNote = (
  metric: string
): { term: string; text: string } | undefined => {
  const d =
    metricDescriptions[metric.toLowerCase() as keyof typeof metricDescriptions];
  return d && "tooltip" in d
    ? { term: `${metric} — ${d.short}`, text: d.tooltip }
    : undefined;
};

export const metricDescriptions = {
  ttfa: {
    short: "Time to First Audio",
    detailed:
      "Delivering natural and responsive voice agents requires both speed and consistency. At Coval, we understand that latency is critical for realistic conversations, which is why we go beyond average measurements to track comprehensive percentile metrics with continuous 15-minute evaluation cycles. This rigorous approach ensures your voice AI maintains the reliable performance necessary for engaging user experiences."
  },
  ttft: {
    short: "Time to First Token",
    tooltip:
      "Time until the provider returns its first transcript token, showing how quickly partial results start streaming. Low TTFT with high TTFS points to slow finalization. Lower is better.",
    detailed:
      "We run TTFT measurements on a fixed test case to measure consistency over time. A model that consistently responds within a narrow time range provides better user experience than one with highly variable timing, even if the variable model is sometimes faster. Unpredictable delays can be seen through sudden latency spikes."
  },
  ttfs: {
    short: "Time to Final Segment",
    tooltip:
      "Time from end of speech until the final transcript is returned, the delay a voice agent actually waits before it can respond. Lower is better.",
    detailed:
      "TTFS measures how quickly a provider returns the final transcript once speech has ended, anchored at a shared VAD end-of-speech point so every provider is compared from the same instant. It isolates engine finalization speed, the latency a voice agent actually waits on before it can respond, independent of how long the speaker talked."
  },
  wer: {
    short: "Word Error Rate (%)",
    detailed:
      "Ensuring accurate speech output is fundamental to user trust and comprehension in voice AI systems. We recognize that even minor pronunciation errors can undermine the entire conversation experience and our evaluation captures how faithfully text-to-speech systems pronounce complex terminology, proper nouns, and domain-specific vocabulary that matter most to your users."
  },
  pareto: {
    short: "Pareto frontier",
    tooltip:
      "The dashed line traces the best WER you can get at each latency budget. Bright dots earn their spot: nothing beats them on speed and accuracy at once. Every faded dot loses on both counts to some bright one.",
    detailed:
      "Speed and accuracy pull against each other, and the dashed line shows what the trade actually costs: follow it to read the best WER on offer at each latency budget. Bright models set that boundary — nothing beats them on both axes at once. Faded models are beaten outright by a bright one, so picking them only makes sense for reasons this chart can't see, like price or language coverage."
  },
  // Shared by the Latency Variation card's description and its headline tooltip.
  iqr: {
    short: "Interquartile Range",
    tooltip:
      "The width of the middle 50% of runs, p75 − p25 — the box drawn for each model. Narrow means predictable; wide means erratic even when the average looks good. Lower is better.",
    detailed:
      "IQR is the width of the middle 50% of runs, p75 − p25 — the box drawn for each model. Narrow distributions mean reliable, predictable response times; wide ones mean erratic performance despite good average speeds, so a moderate median with a tight box often beats a faster median with high variability. The headline averages it across the models shown."
  },
  v2v: {
    short: "Voice-to-Voice Latency",
    detailed:
      "V2V latency measures the time from the end of the user's speech (the end of the last transcribed word) to the first frame of the agent's audio response, measured directly from the conversation audio on native single-turn interactions. Because it is derived from the recorded audio rather than internal events, it reflects the full response time a caller actually experiences — including any pipeline overhead — which makes it a fair cross-model comparison of conversational responsiveness."
  }
};
