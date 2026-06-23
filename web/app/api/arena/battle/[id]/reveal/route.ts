export const runtime = "nodejs";

import { arenaRunnerFetch } from "@/lib/arena/runner";
import type { Reveal } from "@/lib/arena/types";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const res = await arenaRunnerFetch(`/v1/arena/battle/${encodeURIComponent(id)}/reveal`);
  if (!res.ok) return Response.json({ error: "Reveal failed." }, { status: res.status });
  return Response.json((await res.json()) as Reveal);
}
