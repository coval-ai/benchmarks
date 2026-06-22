import { getEnabledTtsModels } from "../playground/providers";
import type { BattleSource, BlindBattle, Reveal, RevealedModel, VoteInput, VoteResult } from "./types";

// Self-contained mock: no network, no backend. Generates two distinct audible tones as
// WAV data URIs so the real <audio> player works exactly as in production, picks two
// random models for the blind A/B, and remembers the assignment in-memory for the reveal.

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function base64FromBuffer(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function toneDataUri(freq: number, seconds = 1.8): string {
  const sampleRate = 8000;
  const n = Math.floor(sampleRate * seconds);
  const dataSize = n * 2;
  const view = new DataView(new ArrayBuffer(44 + dataSize));
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  const amp = 0.25 * 32767;
  for (let i = 0; i < n; i++) {
    const env = Math.min(1, i / 400, (n - i) / 400); // fade edges to avoid clicks
    view.setInt16(44 + i * 2, Math.sin((2 * Math.PI * freq * i) / sampleRate) * amp * env, true);
  }
  return `data:audio/wav;base64,${base64FromBuffer(view.buffer)}`;
}

function toRevealed(m: { provider: string; model: string; label: string }): RevealedModel {
  return { provider: m.provider, model: m.model, label: m.label };
}

export class MockBattleSource implements BattleSource {
  private assignments = new Map<string, Reveal>();
  private counter = 0;

  async createBattle(text: string): Promise<BlindBattle> {
    await delay(600); // exercise the loading state
    const [a, b] = [...getEnabledTtsModels()].sort(() => Math.random() - 0.5);
    if (!a || !b) throw new Error("arena mock: need at least 2 enabled TTS models");
    const battleId = `mock-${++this.counter}-${Math.random().toString(36).slice(2, 8)}`;
    this.assignments.set(battleId, { a: toRevealed(a), b: toRevealed(b) });
    return {
      battleId,
      prompt: text,
      audioA: toneDataUri(220), // A3
      audioB: toneDataUri(330), // E4 — clearly distinct from A
    };
  }

  async submitVote(input: VoteInput): Promise<VoteResult> {
    await delay(200);
    return { battleId: input.battleId, outcome: input.outcome };
  }

  async reveal(battleId: string): Promise<Reveal> {
    const r = this.assignments.get(battleId);
    if (!r) throw new Error(`mock reveal: unknown battle ${battleId}`);
    return r;
  }
}
