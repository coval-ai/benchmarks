// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { ARENA_DOMAINS, isArenaDomain } from "./domains";
import { MockBattleSource } from "./mockSource";
import { getBattleSource } from "./source";

describe("arena domains", () => {
  it("accepts every dropdown value and rejects everything else", () => {
    for (const d of ARENA_DOMAINS) expect(isArenaDomain(d.value)).toBe(true);
    expect(isArenaDomain("all")).toBe(false);
    expect(isArenaDomain("")).toBe(false);
    expect(isArenaDomain(undefined)).toBe(false);
  });
});

describe("MockBattleSource", () => {
  it("creates a blind battle: prompt echoed, two distinct playable audio sources", async () => {
    const battle = await new MockBattleSource().createBattle("hello world", "healthcare");
    expect(battle.battleId).toBeTruthy();
    expect(battle.prompt).toBe("hello world");
    expect(battle.audioA).toMatch(/^data:audio\/wav;base64,/);
    expect(battle.audioB).toMatch(/^data:audio\/wav;base64,/);
    expect(battle.audioA).not.toBe(battle.audioB);
  });

  it("does not leak any model identity in the blind battle payload", async () => {
    const battle = await new MockBattleSource().createBattle("hi", "sales");
    expect(Object.keys(battle).sort()).toEqual(["audioA", "audioB", "battleId", "prompt"]);
  });

  it("reveals two distinct models for a created battle", async () => {
    const src = new MockBattleSource();
    const battle = await src.createBattle("hi", "other");
    const reveal = await src.reveal(battle.battleId, "t");
    expect(reveal.a.model).toBeTruthy();
    expect(reveal.b.model).toBeTruthy();
    expect(reveal.a.model).not.toBe(reveal.b.model);
  });

  it("throws when revealing an unknown battle", async () => {
    await expect(new MockBattleSource().reveal("does-not-exist", "t")).rejects.toThrow();
  });

  it("echoes the outcome on submitVote", async () => {
    const src = new MockBattleSource();
    const battle = await src.createBattle("hi", "customer-service");
    const res = await src.submitVote({ battleId: battle.battleId, outcome: "A_WIN", voterId: "t" });
    expect(res).toEqual({ battleId: battle.battleId, outcome: "A_WIN" });
  });

  it("rejects a vote for an unknown battle (mirrors the backend)", async () => {
    const src = new MockBattleSource();
    await expect(
      src.submitVote({ battleId: "does-not-exist", outcome: "A_WIN", voterId: "t" }),
    ).rejects.toThrow();
  });
});

describe("getBattleSource", () => {
  it("defaults to the mock and returns a singleton", () => {
    const first = getBattleSource();
    expect(first).toBeInstanceOf(MockBattleSource);
    expect(getBattleSource()).toBe(first);
  });
});
