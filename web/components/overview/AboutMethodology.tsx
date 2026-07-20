// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState } from "react";

interface MethodologyTerm {
  code: string;
  tag?: string;
  name: string;
  def: string;
}

interface MethodologySection {
  title: string;
  summary: string;
  terms: MethodologyTerm[];
  details: string[];
}

const SECTIONS: MethodologySection[] = [
  {
    title: "Our Datasets",
    summary:
      "Frozen, versioned datasets spanning studio-quality read speech and spontaneous conversational audio.",
    terms: [
      { code: "clean", tag: "STT", name: "Clean read speech", def: "284 FLEURS clips, the undistorted control" },
      { code: "accent", tag: "STT", name: "Accented speech", def: "845 clips spanning demographic accents" },
      { code: "noisegap", tag: "STT", name: "Background noise", def: "284 clips with intermittent noise bursts" },
      { code: "reverb", tag: "STT", name: "Reverberation", def: "284 clips with room echo" },
      { code: "farfield", tag: "STT", name: "Far-field mic", def: "284 clips recorded at distance" },
      { code: "clipping", tag: "STT", name: "Clipped audio", def: "284 clips with peak distortion" },
      { code: "phonecodec", tag: "STT", name: "Phone codec", def: "284 clips through telephony compression" },
      { code: "stt-v3", tag: "STT", name: "Conversational clips", def: "897 spontaneous voice-agent turns from pipecat's stt-benchmark-data" },
      { code: "tts-v1", tag: "TTS", name: "Text prompts", def: "30 short text inputs for synthesis" },
    ],
    details: [
      "Clean studio audio is what most speech-to-text providers train on, so almost everyone scores well on it. We test that too, but we don't stop there. We pull open-source datasets that cover the conditions real calls actually run into: accents, background noise, reverb, mic distance, clipping, and phone-line compression, plus spontaneous conversational turns full of the fragments and fillers of real speech. Every clip is loudness-normalized to −20 dBFS RMS and locked to a SHA-256 hash, so the audio a provider hears stays identical from one run to the next.",
      "Each scheduled run draws a random sample from its manifest and every model scores on that identical draw, so comparisons within a run are apples-to-apples while the full set is covered over time. Retired datasets stay published for historical reproducibility.",
    ],
  },
  {
    title: "How We Calculate Metrics",
    summary:
      "Each board carries the metrics that matter for its modality, reported as percentiles across every evaluation run.",
    terms: [
      { code: "TTFA", tag: "TTS", name: "Time to First Audio", def: "first audio chunk arrival plus any leading silence before the first audible sample" },
      { code: "TTFT", tag: "STT", name: "Time to First Token", def: "how quickly partial transcripts start streaming" },
      { code: "TTFS", tag: "STT", name: "Time to Final Segment", def: "from a shared VAD end-of-speech anchor to the final transcript" },
      { code: "WER", tag: "STT", name: "Word Error Rate", def: "scored after Whisper's EnglishTextNormalizer on both reference and hypothesis" },
      { code: "WER", tag: "TTS", name: "Word Error Rate", def: "synthesized audio transcribed by a fixed ASR model, scored against the input text" },
    ],
    details: [
      "Latency metrics measure what a voice agent actually waits on. TTFA counts the delay a listener would notice, including any silence before the first audible sample, not just the moment the first bytes arrive. TTFS starts every provider from the same end-of-speech instant, picked by one shared voice-activity model, so no one gets an edge from where they decide the speaker stopped.",
      "Text-to-speech gets a WER too: we transcribe each provider's synthesized audio with one fixed ASR model (OpenAI's whisper-1) and score it against the input text, so the number measures whether the speech is intelligible and complete. The transcriber's own mistakes put a small floor under every provider equally, and both directions share one scoring pipeline.",
      "We exclude connection setup (TCP, TLS, and protocol handshakes) the same way for every provider, so the numbers reflect the model and not the network. Aggregates show the full spread from p25 to p99 next to the mean, and leaderboards rank by the median.",
    ],
  },
  {
    title: "Evaluation Standards",
    summary:
      "Pinned inputs, reproducible outputs. Every leaderboard number traces back to the exact configuration that produced it.",
    terms: [
      { code: "model_id", name: "Exact model versions", def: "every provider entry uses the versioned identifier the provider exposes, never an alias" },
      { code: "norm_version", name: "Versioned pipeline", def: "a change to WER methodology never mixes with rows produced before it" },
      { code: "runner_sha", name: "Traceable runs", def: "every run records the runner commit that produced it" },
    ],
    details: [
      "Scoring runs on standard, lock-file-pinned libraries: jiwer for edit distance and whisper-normalizer for text normalization. When we change the methodology, we mark results from before and after as incomparable instead of quietly blending them.",
      "The runner, the dataset manifests, and the methodology docs are all open source, and every audio file is checked against its SHA-256 hash on fetch. Every number on this site can be reproduced straight from the repository.",
    ],
  },
];

const MethodologyRow: React.FC<{ section: MethodologySection }> = ({ section }) => {
  const [open, setOpen] = useState(false);
  const panelId = section.title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  return (
    <div className="border-t border-border-primary last:border-b">
      <h3>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((prev) => !prev)}
          className="group flex w-full items-start justify-between gap-4 py-6 text-left"
        >
          <span>
            <span className="block text-base font-medium text-text-primary md:text-lg">
              {section.title}
            </span>
            <span className="mt-1 block max-w-prose text-sm font-light leading-snug text-text-secondary">
              {section.summary}
            </span>
          </span>
          <span
            aria-hidden
            className="mt-0.5 shrink-0 font-mono text-xl leading-none text-text-tertiary transition-colors group-hover:text-text-primary"
          >
            {open ? "–" : "+"}
          </span>
        </button>
      </h3>
      <div
        id={panelId}
        className={`grid transition-[grid-template-rows] duration-300 ease-in-out ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
      >
        <div className="overflow-hidden">
          <div className="pb-8">
            <dl className="mb-4 divide-y divide-border-secondary overflow-hidden rounded-lg border border-border-secondary bg-surface-elevated">
              {section.terms.map(({ code, tag, name, def }) => (
                <div
                  key={`${code}-${tag ?? ""}`}
                  className="grid grid-cols-[6.75rem_1fr] items-baseline gap-x-4 px-4 py-3 sm:grid-cols-[8.5rem_1fr]"
                >
                  <dt className="flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
                    <span className="font-mono text-xs text-text-primary">{code}</span>
                    {tag && (
                      <span className="rounded bg-surface-secondary px-1 py-px font-mono text-[0.6rem] tracking-wider text-text-tertiary">
                        {tag}
                      </span>
                    )}
                  </dt>
                  <dd className="text-xs leading-relaxed text-text-secondary">
                    <span className="font-medium text-text-primary">{name}</span> &mdash; {def}
                  </dd>
                </div>
              ))}
            </dl>
            {section.details.map((paragraph) => (
              <p
                key={paragraph.slice(0, 24)}
                className="mt-3 max-w-prose text-sm font-light leading-relaxed text-text-secondary"
              >
                {paragraph}
              </p>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const AboutMethodology: React.FC = () => (
  <section
    aria-labelledby="about-methodology"
    className="grid grid-cols-1 gap-10 pb-10 pt-16 md:pt-24 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] lg:gap-16"
  >
    <div className="lg:sticky lg:top-24 lg:self-start">
      <p className="font-mono text-sm text-text-tertiary">Methodology</p>
      <h2
        id="about-methodology"
        className="mt-3 scroll-mt-24 text-balance text-2xl font-medium leading-tight tracking-tight text-text-primary md:text-3xl"
      >
        About Metrics &amp; Methodology
      </h2>
      <p className="mt-3 max-w-md text-pretty text-base font-light leading-snug text-text-secondary">
        Every leaderboard number is a function of pinned inputs. Here is what we
        run, how we score it, and why you can trust it.
      </p>
      <a
        href="https://github.com/coval-ai/benchmarks/blob/main/docs/methodology.md"
        target="_blank"
        rel="noopener noreferrer"
        className="mt-5 inline-flex items-center gap-1 text-sm font-medium text-text-secondary transition-colors hover:text-text-primary"
      >
        Read the full methodology <span aria-hidden>&rarr;</span>
      </a>
    </div>
    <div>
      {SECTIONS.map((section) => (
        <MethodologyRow key={section.title} section={section} />
      ))}
    </div>
  </section>
);

export default AboutMethodology;
