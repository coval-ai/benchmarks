export const runtime = "nodejs";
export const maxDuration = 60;

import { isArenaDomain } from "@/lib/arena/domains";
import { arenaAccessOk } from "@/lib/arena/guard";
import { arenaRunnerFetch } from "@/lib/arena/runner";
import type { BlindBattle } from "@/lib/arena/types";

interface BattleOut {
  id: string;
  prompt_text: string;
  domain: string | null;
  audio_a_url: string;
  audio_b_url: string;
}

export async function POST(req: Request) {
  if (!(await arenaAccessOk())) return new Response(null, { status: 404 });
  let body: { text?: string; domain?: string };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON." }, { status: 400 });
  }
  const text = body.text?.trim();
  if (!text) return Response.json({ error: "Prompt is empty." }, { status: 400 });
  if (!isArenaDomain(body.domain)) {
    return Response.json({ error: "Invalid domain." }, { status: 400 });
  }

  let res: Response;
  try {
    res = await arenaRunnerFetch("/v1/arena/battle", { prompt: text, domain: body.domain });
  } catch {
    return Response.json({ error: "Battle generation failed." }, { status: 502 });
  }
  if (!res.ok) return Response.json({ error: "Battle generation failed." }, { status: res.status });

  const b = (await res.json()) as BattleOut;
  const battle: BlindBattle = {
    battleId: b.id,
    prompt: b.prompt_text,
    audioA: b.audio_a_url,
    audioB: b.audio_b_url,
  };
  return Response.json(battle, { status: 201 });
}
