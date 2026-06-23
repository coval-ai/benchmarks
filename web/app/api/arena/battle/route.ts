export const runtime = "nodejs";
export const maxDuration = 60;

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
  let body: { text?: string };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON." }, { status: 400 });
  }
  const text = body.text?.trim();
  if (!text) return Response.json({ error: "Prompt is empty." }, { status: 400 });

  const res = await arenaRunnerFetch("/v1/arena/battle", { prompt: text });
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
