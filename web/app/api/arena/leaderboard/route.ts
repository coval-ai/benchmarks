export const runtime = "nodejs";

import { arenaAccessOk } from "@/lib/arena/guard";
import { arenaRunnerFetch } from "@/lib/arena/runner";

export async function GET(req: Request) {
  if (!(await arenaAccessOk())) return new Response(null, { status: 404 });
  const { searchParams } = new URL(req.url);
  const qs = new URLSearchParams({
    metric: searchParams.get("metric") ?? "naturalness",
    domain: searchParams.get("domain") ?? "all",
  });

  let res: Response;
  try {
    res = await arenaRunnerFetch(`/v1/arena/leaderboard?${qs}`);
  } catch {
    return Response.json({ error: "Leaderboard unavailable." }, { status: 502 });
  }
  if (!res.ok) return Response.json({ error: "Leaderboard unavailable." }, { status: res.status });
  return Response.json(await res.json());
}
