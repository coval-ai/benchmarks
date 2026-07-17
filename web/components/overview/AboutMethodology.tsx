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
      { code: "stt-v1", tag: "STT", name: "LibriSpeech test-clean", def: "50 studio-quality read utterances (CC-BY-4.0), the easy tier" },
      { code: "stt-v3", tag: "STT", name: "Conversational clips", def: "897 spontaneous voice-agent turns from pipecat's stt-benchmark-data, the hard tier" },
      { code: "tts-v1", tag: "TTS", name: "Text prompts", def: "30 short text inputs for synthesis" },
    ],
    details: [
      "Speech-to-text runs both tiers each cycle: studio-quality read speech that most providers train on, and real conversational speech — fragments, fillers, prompted turns. Every clip is loudness-normalized to −20 dBFS RMS and pinned by SHA-256, so the audio a provider hears is byte-identical across runs.",
      "Each scheduled run draws a random sample from its manifest and every model scores on that identical draw, so comparisons within a run are apples-to-apples while the full set is covered over time. Retired datasets stay published for historical reproducibility.",
    ],
  },
  {
    title: "How We Calculate Metrics",
    summary:
      "Each board carries the metrics that matter for its modality, aggregated as percentiles over continuous evaluation cycles.",
    terms: [
      { code: "TTFA", tag: "TTS", name: "Time to First Audio", def: "first audio chunk arrival plus any leading silence before the first audible sample" },
      { code: "TTFT", tag: "STT", name: "Time to First Token", def: "how quickly partial transcripts start streaming" },
      { code: "TTFS", tag: "STT", name: "Time to Final Segment", def: "from a shared VAD end-of-speech anchor to the final transcript" },
      { code: "WER", tag: "STT", name: "Word Error Rate", def: "scored after Whisper's EnglishTextNormalizer on both reference and hypothesis" },
      { code: "V2V", tag: "S2S", name: "Voice-to-Voice", def: "end of user speech to the first frame of agent audio, from the recorded conversation" },
    ],
    details: [
      "Latency metrics measure what a voice agent actually waits on. TTFA is perceived first-audible latency, not just network arrival; TTFS anchors every provider at the same VAD end-of-speech instant; V2V is derived from the conversation audio itself, so it includes full pipeline overhead.",
      "Connection setup — TCP, TLS, protocol handshakes — is excluded uniformly across every provider. Aggregates expose the full percentile distribution (p25 through p99) alongside the mean; leaderboards rank by the median (p50).",
    ],
  },
  {
    title: "Evaluation Standards",
    summary:
      "Pinned inputs, reproducible outputs — every leaderboard number traces back to its exact configuration.",
    terms: [
      { code: "matrix", name: "Exact model versions", def: "every provider entry uses the versioned identifier the provider exposes, never an alias" },
      { code: "norm_version", name: "Versioned pipeline", def: "changed WER methodology never mixes with rows produced before it" },
      { code: "runner_sha", name: "Traceable runs", def: "every run records the runner commit that produced it" },
    ],
    details: [
      "Scoring is delegated to standard, lock-file-pinned libraries — jiwer for edit distance, whisper-normalizer for text normalization. When the methodology changes, results from before and after are marked incomparable rather than silently blended.",
      "The full runner, dataset manifests, and methodology docs are open source, and every audio file is integrity-checked by SHA-256 on fetch — every number on this site can be reproduced from the repository.",
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
                  key={code}
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
        className="mt-3 text-balance text-2xl font-medium leading-tight tracking-tight text-text-primary md:text-3xl"
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
