// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { getEnabledSttModels, getEnabledTtsModels } from "./providers";

const RUNNER_STT_INIT = resolve(__dirname, "../../../runner/src/coval_bench/providers/stt/__init__.py");
const RUNNER_TTS_INIT = resolve(__dirname, "../../../runner/src/coval_bench/providers/tts/__init__.py");

function extractRegistryKeys(filePath: string, registryName: string): Set<string> {
  const text = readFileSync(filePath, "utf8");
  const dictPattern = new RegExp(
    String.raw`${registryName}\s*(?::\s*dict\[[^=]+\])?\s*=\s*\{([\s\S]*?)\}`,
    "m",
  );
  const match = dictPattern.exec(text);
  if (!match) throw new Error(`${registryName} dict not found in ${filePath}`);
  const body = match[1] ?? "";
  const keyPattern = /"([^"]+)"\s*:/g;
  const keys = new Set<string>();
  for (const m of body.matchAll(keyPattern)) {
    keys.add(m[1]!);
  }
  const optionalPattern = new RegExp(
    String.raw`${registryName}\[\s*"([^"]+)"\s*\]\s*=`,
    "g",
  );
  for (const m of text.matchAll(optionalPattern)) {
    keys.add(m[1]!);
  }
  return keys;
}

describe("playground provider parity with runner", () => {
  it("every enabled STT provider has a matching runner registry entry", () => {
    const runnerProviders = extractRegistryKeys(RUNNER_STT_INIT, "STT_PROVIDERS");
    expect(runnerProviders.size).toBeGreaterThan(0);
    const playgroundProviders = new Set(getEnabledSttModels().map((m) => m.provider));
    for (const provider of playgroundProviders) {
      expect(
        runnerProviders.has(provider),
        `playground STT provider "${provider}" not registered in runner STT_PROVIDERS`,
      ).toBe(true);
    }
  });

  it("every enabled TTS provider has a matching runner registry entry", () => {
    const runnerProviders = extractRegistryKeys(RUNNER_TTS_INIT, "TTS_PROVIDERS");
    expect(runnerProviders.size).toBeGreaterThan(0);
    const playgroundProviders = new Set(getEnabledTtsModels().map((m) => m.provider));
    for (const provider of playgroundProviders) {
      expect(
        runnerProviders.has(provider),
        `playground TTS provider "${provider}" not registered in runner TTS_PROVIDERS`,
      ).toBe(true);
    }
  });
});
