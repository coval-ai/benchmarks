export const runtime = "nodejs";

import { arenaRunnerFetch } from "@/lib/arena/runner";

export async function GET(req: Request) {
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
