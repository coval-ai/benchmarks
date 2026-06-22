export type Outcome = "A_WIN" | "B_WIN" | "TIE";

// What the UI renders during the blind phase — never any model identity.
export interface BlindBattle {
  battleId: string;
  prompt: string;
  audioA: string; // playable by an <audio> element (real URL, or data URI in the mock)
  audioB: string;
}

export interface VoteInput {
  battleId: string;
  outcome: Outcome;
  voterId: string;
}

export interface VoteResult {
  battleId: string;
  outcome: Outcome;
}

// Returned only after the vote lands — de-anonymizes the two cards.
export interface RevealedModel {
  provider: string;
  model: string;
  label: string;
}

export interface Reveal {
  a: RevealedModel;
  b: RevealedModel;
}

// The single seam the UI depends on. Swap the implementation (mock <-> api) and the
// whole battle page keeps working unchanged. See ./README.md.
export interface BattleSource {
  createBattle(text: string): Promise<BlindBattle>;
  submitVote(input: VoteInput): Promise<VoteResult>;
  reveal(battleId: string): Promise<Reveal>;
}
