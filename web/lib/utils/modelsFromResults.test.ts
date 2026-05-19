// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { Result } from "../aggregates";
import { buildModelsByProviderFromResults } from "./modelsFromResults";

function row(
  provider: string,
  model: string,
  benchmark: "STT" | "TTS" = "STT"
): Result {
  return {
    id: 1,
    provider,
    model,
    benchmark,
    metric_type: "TTFT",
    metric_value: 0.5,
    metric_units: "seconds",
    status: "SUCCEEDED",
    created_at: "2026-05-18T12:00:00Z",
    run_id: 1,
    audio_filename: "a.wav",
    transcript: null,
    error: null,
    voice: null,
  } as Result;
}

describe("buildModelsByProviderFromResults", () => {
  it("includes providers/models present in result rows when no catalogue is available", () => {
    const out = buildModelsByProviderFromResults(
      [
        row("deepgram", "nova-2"),
        row("xai", "grok-stt"),
      ],
      "STT"
    );
    expect(out.deepgram).toEqual(["deepgram:nova-2"]);
    expect(out.xai).toEqual(["xai:grok-stt"]);
    expect(out.openai).toBeUndefined();
  });

  it("starts from enabled catalogue models so an empty benchmark page still has choices", () => {
    const catalogue = {
      tts: [],
      stt: [
        {
          provider: "deepgram",
          models: [
            { model: "nova-2", disabled: false },
            { model: "flux-general-multi", disabled: false },
          ],
        },
      ],
    };
    const out = buildModelsByProviderFromResults(
      [row("deepgram", "nova-2")],
      "STT",
      catalogue as never
    );
    expect(out.deepgram).toEqual([
      "deepgram:nova-2",
      "deepgram:flux-general-multi",
    ]);
  });

  it("works when catalogue is undefined", () => {
    const out = buildModelsByProviderFromResults([row("hume", "octave-2", "TTS")], "TTS");
    expect(out.hume).toEqual(["hume:octave-2"]);
  });

  it("does not include disabled catalogue models or disabled result rows", () => {
    const catalogue = {
      tts: [],
      stt: [
        {
          provider: "openai",
          models: [
            { model: "gpt-realtime-whisper", disabled: false },
            { model: "legacy-whisper", disabled: true },
          ],
        },
      ],
    };

    const out = buildModelsByProviderFromResults(
      [
        row("openai", "gpt-realtime-whisper"),
        row("openai", "legacy-whisper"),
      ],
      "STT",
      catalogue as never
    );

    expect(out.openai).toEqual(["openai:gpt-realtime-whisper"]);
  });

  it("adds result-backed TTS models alongside catalogue models", () => {
    const catalogue = {
      stt: [],
      tts: [
        {
          provider: "openai",
          models: [{ model: "tts-1-hd", disabled: false }],
        },
        {
          provider: "elevenlabs",
          models: [{ model: "eleven_flash_v2_5", disabled: false }],
        },
      ],
    };

    const out = buildModelsByProviderFromResults(
      [
        row("openai", "tts-1-hd", "TTS"),
        row("hume", "octave-tts", "TTS"),
        row("hume", "octave-2", "TTS"),
      ],
      "TTS",
      catalogue as never
    );

    expect(out.openai).toEqual(["openai:tts-1-hd"]);
    expect(out.elevenlabs).toEqual(["elevenlabs:eleven_flash_v2_5"]);
    expect(out.hume).toEqual(["hume:octave-tts", "hume:octave-2"]);
  });
});
