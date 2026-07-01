export const runtime = "nodejs";

import { arenaAccessOk } from "@/lib/arena/guard";
import { arenaRunnerFetch } from "@/lib/arena/runner";
import type { Reveal } from "@/lib/arena/types";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!(await arenaAccessOk())) return new Response(null, { status: 404 });
  const { id } = await params;
  const voterId = new URL(req.url).searchParams.get("voterId");
  if (!voterId) return Response.json({ error: "voterId required." }, { status: 400 });
  let res: Response;
  try {
    res = await arenaRunnerFetch(
      `/v1/arena/battle/${encodeURIComponent(id)}/reveal?voter_id=${encodeURIComponent(voterId)}`,
    );
  } catch {
    return Response.json({ error: "Reveal failed." }, { status: 502 });
  }
  if (!res.ok) return Response.json({ error: "Reveal failed." }, { status: res.status });
  return Response.json((await res.json()) as Reveal);
}
