import type { ArenaDomain } from "./domains";
import type { BattleSource, BlindBattle, Reveal, VoteInput, VoteResult } from "./types";

// Real battle source: talks to same-origin Next.js API proxy routes under /api/arena/*,
// which inject the server-only X-Labeler-Key and proxy to the runner. Inert until
// NEXT_PUBLIC_ARENA_SOURCE=api is set (the factory defaults to the mock). The BFF routes it
// calls:
//   POST /api/arena/battle              { text, domain }                -> BlindBattle
//   POST /api/arena/vote                { battleId, outcome, voterId }  -> VoteResult
//   GET  /api/arena/battle/{id}/reveal?voterId=...                      -> Reveal

// Coarse hang-backstops, not latency SLAs; tune once real backend latency is known.
const QUICK_TIMEOUT_MS = 5_000; // vote/reveal: quick DB ops
const SYNTH_TIMEOUT_MS = 30_000; // createBattle: on-demand TTS synthesis of two clips

async function postJson<T>(url: string, body: unknown, timeoutMs = QUICK_TIMEOUT_MS): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return (await res.json()) as T;
}

export class ApiBattleSource implements BattleSource {
  createBattle(text: string, domain: ArenaDomain): Promise<BlindBattle> {
    return postJson<BlindBattle>("/api/arena/battle", { text, domain }, SYNTH_TIMEOUT_MS);
  }

  submitVote(input: VoteInput): Promise<VoteResult> {
    return postJson<VoteResult>("/api/arena/vote", input);
  }

  async reveal(battleId: string, voterId: string): Promise<Reveal> {
    const url = `/api/arena/battle/${encodeURIComponent(battleId)}/reveal?voterId=${encodeURIComponent(voterId)}`;
    const res = await fetch(url, {
      signal: AbortSignal.timeout(QUICK_TIMEOUT_MS),
    });
    if (!res.ok) throw new Error(`reveal -> ${res.status}`);
    return (await res.json()) as Reveal;
  }
}
