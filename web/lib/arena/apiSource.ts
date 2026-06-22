import type { BattleSource, BlindBattle, Reveal, VoteInput, VoteResult } from "./types";

// Real battle source: talks to same-origin Next.js API proxy routes under /api/arena/*,
// which inject the server-only X-Labeler-Key and proxy to the runner. Inert until
// NEXT_PUBLIC_ARENA_SOURCE=api is set (the factory defaults to the mock). The BFF routes and
// the runner endpoints they call are not built yet — this adapter defines the contract they
// must satisfy:
//   POST /api/arena/battle              { text }                        -> BlindBattle
//   POST /api/arena/vote                { battleId, outcome, voterId }  -> VoteResult
//   GET  /api/arena/battle/{id}/reveal                                  -> Reveal

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return (await res.json()) as T;
}

export class ApiBattleSource implements BattleSource {
  createBattle(text: string): Promise<BlindBattle> {
    return postJson<BlindBattle>("/api/arena/battle", { text });
  }

  submitVote(input: VoteInput): Promise<VoteResult> {
    return postJson<VoteResult>("/api/arena/vote", input);
  }

  async reveal(battleId: string): Promise<Reveal> {
    const res = await fetch(`/api/arena/battle/${encodeURIComponent(battleId)}/reveal`);
    if (!res.ok) throw new Error(`reveal -> ${res.status}`);
    return (await res.json()) as Reveal;
  }
}
