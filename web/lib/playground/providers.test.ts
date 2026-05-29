// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { getEnabledSttModels, getEnabledTtsModels } from "./providers";

const RUNNER_STT_DIR = resolve(__dirname, "../../../runner/src/coval_bench/providers/stt");
const RUNNER_TTS_DIR = resolve(__dirname, "../../../runner/src/coval_bench/providers/tts");
const RUNNER_STT_INIT = resolve(RUNNER_STT_DIR, "__init__.py");
const RUNNER_TTS_INIT = resolve(RUNNER_TTS_DIR, "__init__.py");

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

function extractRunnerModels(providerFile: string): Set<string> | null {
  const text = readFileSync(providerFile, "utf8");
  const decl = /_VALID_MODELS\s*=\s*frozenset\(([\s\S]*?)\)/m.exec(text);
  if (!decl) return null;
  let body = decl[1] ?? "";
  const ref = /^\s*([A-Za-z_]\w*)\s*$/.exec(body);
  if (ref) {
    const def = new RegExp(String.raw`${ref[1]}\s*(?::[^=\n]+)?=\s*\{([\s\S]*?)\}`, "m").exec(text);
    body = def ? (def[1] ?? "") : "";
  }
  const models = new Set<string>();
  for (const m of body.matchAll(/"([^"]+)"\s*:/g)) models.add(m[1]!);
  if (models.size === 0) {
    for (const m of body.matchAll(/"([^"]+)"/g)) models.add(m[1]!);
  }
  return models;
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

  it("every enabled STT model exists in the runner provider's _VALID_MODELS", () => {
    const skipped: string[] = [];
    for (const m of getEnabledSttModels()) {
      const models = extractRunnerModels(resolve(RUNNER_STT_DIR, `${m.provider}.py`));
      if (!models) {
        skipped.push(`${m.provider}:${m.model}`);
        continue;
      }
      expect(
        models.has(m.model),
        `playground STT model "${m.model}" (${m.id}) not in runner ${m.provider} _VALID_MODELS`,
      ).toBe(true);
    }
    if (skipped.length > 0) {
      console.warn(`[parity] STT model check skipped (no static allowlist): ${skipped.join(", ")}`);
    }
  });

  it("every enabled TTS model exists in the runner provider's _VALID_MODELS", () => {
    const skipped: string[] = [];
    for (const m of getEnabledTtsModels()) {
      const models = extractRunnerModels(resolve(RUNNER_TTS_DIR, `${m.provider}.py`));
      if (!models) {
        skipped.push(`${m.provider}:${m.model}`);
        continue;
      }
      expect(
        models.has(m.model),
        `playground TTS model "${m.model}" (${m.id}) not in runner ${m.provider} _VALID_MODELS`,
      ).toBe(true);
    }
    if (skipped.length > 0) {
      console.warn(`[parity] TTS model check skipped (no static allowlist): ${skipped.join(", ")}`);
    }
  });
});
