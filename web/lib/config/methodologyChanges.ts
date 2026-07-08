// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export type MethodologyMetricKey = "ttfa" | "ttft" | "ttfs" | "wer";

export interface MethodologyChange {
  date: string;
  time?: string;
  metrics?: MethodologyMetricKey[];
  title: string;
  detail: string;
}

export const methodologyChanges: MethodologyChange[] = [
  {
    date: "2026-06-01",
    metrics: ["ttfa"],
    title: "TTFA redefined as perceived first-audible latency",
    detail:
      "TTFA previously measured network arrival only (time to the first audio chunk). It now adds the leading silence a provider front-loads before the first audible sample, so it reflects what a listener actually waits to hear. Every provider's TTFA shifts upward; values before and after are not comparable.",
  },
  {
    date: "2026-06-03",
    metrics: ["ttfa"],
    title: "TTFA corrected for librosa cold-start",
    detail:
      "Early TTFA readings were inflated because the audio library loaded lazily on the first request, adding one-off import latency to the measurement. The library is now warmed before traffic, so TTFA reflects real synthesis latency. Values drop to their corrected level after this date.",
  },
  {
    date: "2026-06-10",
    metrics: ["ttfs"],
    title: "TTFS anchored at end-of-speech",
    detail:
      "Streaming STT finalization is now forced at a shared VAD end-of-speech point, so every provider is timed from the same instant. Steps after this date reflect engine finalization speed, not how long the speaker talked.",
  },
  {
    date: "2026-06-15",
    metrics: ["ttfs"],
    title: "Gradium finalization made event-driven",
    detail:
      "Gradium STT now waits for an explicit finalization event instead of a fixed timeout, so its time-to-final-segment reflects true engine speed. Gradium TTFS values shift at this point.",
  },
  {
    date: "2026-07-01",
    time: "14:00:00-07:00",
    metrics: ["ttft", "ttfs"],
    title: "STT streaming latency corrected for pacing drift",
    detail:
      "Streaming STT audio was fed with a fixed per-chunk sleep, so send-loop drift accumulated and inflated latency. Audio is now paced against an absolute deadline, so latency reflects true engine speed. STT latency drops after this date; values before and after are not comparable.",
  },
  {
    date: "2026-07-07",
    time: "17:33:45-07:00",
    metrics: ["wer"],
    title: "WER normalization switched to the Whisper text normalizer",
    detail:
      "Transcripts were previously normalized with a hand-rolled pipeline that corrupted many number forms (“thirty six” became 3006), putting a WER floor under providers that transcribe numbers as digits. Both reference and hypothesis now go through OpenAI’s Whisper English text normalizer, the standard used for published WER. Affected providers’ WER drops after this date; values before and after are not comparable.",
  },
  {
    date: "2026-07-08",
    time: "09:52:07-07:00",
    metrics: ["wer", "ttft", "ttfs"],
    title: "STT dataset replaced with WildASR fleurs_clean_en",
    detail:
      "The STT benchmark corpus moved from LibriSpeech test-clean to the clean English FLEURS subset of WildASR (stt-v2), loudness-normalized to a −20 dBFS RMS target. LibriSpeech is heavily represented in provider training data, which compressed WER differences between engines. WER values before and after are not comparable; latency metrics shift with the new clip mix. The runner also now streams exact PCM frames to STT providers (previously the WAV container header was included at stream start).",
  },
];
