// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

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
    res = await arenaRunnerFetch(`/v1/arena/admin/convergence?${qs}`);
  } catch {
    return new Response("Report unavailable.", { status: 502 });
  }
  if (!res.ok) return new Response("Report unavailable.", { status: res.status });
  return new Response(await res.text(), {
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}
