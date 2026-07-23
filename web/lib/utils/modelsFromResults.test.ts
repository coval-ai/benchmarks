// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { ProviderModelRef } from "./modelsFromResults";
import { buildModelsByProvider } from "./modelsFromResults";

function entry(provider: string, model: string): ProviderModelRef {
  return { provider, model };
}

describe("buildModelsByProvider", () => {
  it("includes providers/models present in entries when no catalogue is available", () => {
    const out = buildModelsByProvider(
      [
        entry("deepgram", "nova-2"),
        entry("xai", "grok-stt"),
      ],
      "STT"
    );
    expect(out.deepgram).toEqual(["deepgram:nova-2"]);
    expect(out.xai).toEqual(["xai:grok-stt"]);
    expect(out.openai).toBeUndefined();
  });

  it("keeps same-slug models from different providers distinct", () => {
    const out = buildModelsByProvider(
      [entry("speechmatics", "default"), entry("gradium", "default")],
      "STT"
    );
    expect(out.speechmatics).toEqual(["speechmatics:default"]);
    expect(out.gradium).toEqual(["gradium:default"]);
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
    const out = buildModelsByProvider(
      [entry("deepgram", "nova-2")],
      "STT",
      catalogue as never
    );
    expect(out.deepgram).toEqual([
      "deepgram:nova-2",
      "deepgram:flux-general-multi",
    ]);
  });

  it("reads the s2s catalogue for the S2S benchmark", () => {
    const catalogue = {
      stt: [],
      tts: [],
      s2s: [
        { provider: "openai", models: [{ model: "gpt-realtime", disabled: false }] },
        { provider: "google", models: [{ model: "gemini-live", disabled: false }] },
      ],
    };
    const out = buildModelsByProvider(
      [entry("openai", "gpt-realtime")],
      "S2S",
      catalogue as never
    );
    expect(out.openai).toEqual(["openai:gpt-realtime"]);
    expect(out.google).toEqual(["google:gemini-live"]);
  });

  it("works when catalogue is undefined", () => {
    const out = buildModelsByProvider([entry("hume", "octave-2")], "TTS");
    expect(out.hume).toEqual(["hume:octave-2"]);
  });

  it("does not include disabled catalogue models or disabled data-backed entries", () => {
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

    const out = buildModelsByProvider(
      [
        entry("openai", "gpt-realtime-whisper"),
        entry("openai", "legacy-whisper"),
      ],
      "STT",
      catalogue as never
    );

    expect(out.openai).toEqual(["openai:gpt-realtime-whisper"]);
  });

  it("does not include dedicated STT or TTS models even when aggregate entries exist", () => {
    const dedicatedTag = {
      category: "source",
      value: "dedicated-inference",
      label: "Dedicated inference",
    };
    const catalogue = {
      tts: [
        {
          provider: "baseten",
          models: [
            {
              model: "qwen3-tts-1.7b",
              disabled: false,
              tags: [dedicatedTag],
            },
          ],
        },
        {
          provider: "openai",
          models: [{ model: "gpt-4o-mini-tts", disabled: false }],
        },
      ],
      stt: [
        {
          provider: "baseten",
          models: [
            {
              model: "whisper-large-v3",
              disabled: false,
              tags: [dedicatedTag],
            },
          ],
        },
        {
          provider: "deepgram",
          models: [{ model: "nova-3", disabled: false }],
        },
      ],
      s2s: [],
    };

    const tts = buildModelsByProvider(
      [
        entry("baseten", "qwen3-tts-1.7b"),
        entry("openai", "gpt-4o-mini-tts"),
      ],
      "TTS",
      catalogue as never
    );

    const stt = buildModelsByProvider(
      [
        entry("baseten", "whisper-large-v3"),
        entry("deepgram", "nova-3"),
      ],
      "STT",
      catalogue as never
    );

    expect(tts.baseten).toBeUndefined();
    expect(tts.openai).toEqual(["openai:gpt-4o-mini-tts"]);
    expect(stt.baseten).toBeUndefined();
    expect(stt.deepgram).toEqual(["deepgram:nova-3"]);
  });

  it("adds data-backed TTS models alongside catalogue models", () => {
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

    const out = buildModelsByProvider(
      [
        entry("openai", "tts-1-hd"),
        entry("hume", "octave-tts"),
        entry("hume", "octave-2"),
      ],
      "TTS",
      catalogue as never
    );

    expect(out.openai).toEqual(["openai:tts-1-hd"]);
    expect(out.elevenlabs).toEqual(["elevenlabs:eleven_flash_v2_5"]);
    expect(out.hume).toEqual(["hume:octave-tts", "hume:octave-2"]);
  });

  it("does not deduplicate repeated entries into duplicate keys", () => {
    const out = buildModelsByProvider(
      [
        entry("deepgram", "nova-2"),
        entry("deepgram", "nova-2"),
        entry("deepgram", "nova-2"),
      ],
      "STT"
    );
    expect(out.deepgram).toEqual(["deepgram:nova-2"]);
  });
});
