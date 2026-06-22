// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { MockBattleSource } from "./mockSource";
import { getBattleSource } from "./source";

describe("MockBattleSource", () => {
  it("creates a blind battle: prompt echoed, two distinct playable audio sources", async () => {
    const battle = await new MockBattleSource().createBattle("hello world");
    expect(battle.battleId).toBeTruthy();
    expect(battle.prompt).toBe("hello world");
    expect(battle.audioA).toMatch(/^data:audio\/wav;base64,/);
    expect(battle.audioB).toMatch(/^data:audio\/wav;base64,/);
    expect(battle.audioA).not.toBe(battle.audioB);
  });

  it("does not leak any model identity in the blind battle payload", async () => {
    const battle = await new MockBattleSource().createBattle("hi");
    expect(Object.keys(battle).sort()).toEqual(["audioA", "audioB", "battleId", "prompt"]);
  });

  it("reveals two distinct models for a created battle", async () => {
    const src = new MockBattleSource();
    const battle = await src.createBattle("hi");
    const reveal = await src.reveal(battle.battleId);
    expect(reveal.a.model).toBeTruthy();
    expect(reveal.b.model).toBeTruthy();
    expect(reveal.a.model).not.toBe(reveal.b.model);
  });

  it("throws when revealing an unknown battle", async () => {
    await expect(new MockBattleSource().reveal("does-not-exist")).rejects.toThrow();
  });

  it("echoes the outcome on submitVote", async () => {
    const src = new MockBattleSource();
    const battle = await src.createBattle("hi");
    const res = await src.submitVote({ battleId: battle.battleId, outcome: "A_WIN", voterId: "t" });
    expect(res).toEqual({ battleId: battle.battleId, outcome: "A_WIN" });
  });
});

describe("getBattleSource", () => {
  it("defaults to the mock and returns a singleton", () => {
    const first = getBattleSource();
    expect(first).toBeInstanceOf(MockBattleSource);
    expect(getBattleSource()).toBe(first);
  });
});
