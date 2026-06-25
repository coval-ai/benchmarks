export const runtime = "nodejs";

import { arenaAccessOk } from "@/lib/arena/guard";
import { arenaRunnerFetch } from "@/lib/arena/runner";
import type { VoteResult } from "@/lib/arena/types";

interface VoteOut {
  battle_id: string;
  outcome: string;
}

export async function POST(req: Request) {
  if (!(await arenaAccessOk())) return new Response(null, { status: 404 });
  let body: { battleId?: string; outcome?: string; voterId?: string };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON." }, { status: 400 });
  }
  const { battleId, outcome, voterId } = body;
  if (!battleId || !outcome || !voterId) {
    return Response.json({ error: "battleId, outcome, voterId required." }, { status: 400 });
  }

  let res: Response;
  try {
    res = await arenaRunnerFetch("/v1/arena/vote", {
      battle_id: battleId,
      outcome,
      voter_id: voterId,
    });
  } catch {
    return Response.json({ error: "Vote failed." }, { status: 502 });
  }
  if (!res.ok) return Response.json({ error: "Vote failed." }, { status: res.status });

  const v = (await res.json()) as VoteOut;
  const result: VoteResult = { battleId: v.battle_id, outcome: v.outcome as VoteResult["outcome"] };
  return Response.json(result, { status: 201 });
}
