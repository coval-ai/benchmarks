import { ApiBattleSource } from "./apiSource";
import { MockBattleSource } from "./mockSource";
import type { BattleSource } from "./types";

// The UI imports ONLY this. Flip NEXT_PUBLIC_ARENA_SOURCE=api to go live; default is the
// self-contained mock (no backend required). Singleton so the mock's in-memory battle
// assignments survive across calls within a session (needed for reveal()).
let instance: BattleSource | null = null;

export function getBattleSource(): BattleSource {
  if (instance) return instance;
  instance =
    process.env.NEXT_PUBLIC_ARENA_SOURCE === "api" ? new ApiBattleSource() : new MockBattleSource();
  return instance;
}

export type { BattleSource, BlindBattle, Outcome, Reveal, VoteInput, VoteResult } from "./types";
