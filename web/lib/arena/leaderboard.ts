// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from "@tanstack/react-query";

export type ArenaStatus = "preliminary" | "usable" | "established";

export interface ArenaLeaderboardEntry {
  provider: string;
  model: string;
  rating_elo: number;
  rating_bt: number;
  ci_low: number | null;
  ci_high: number | null;
  ci_half_width: number | null;
  votes_total: number;
  wins: number;
  losses: number;
  ties: number;
  status: ArenaStatus;
}

export interface ArenaLeaderboard {
  metric: string;
  domain: string;
  computed_at: string | null;
  methodology_version: string | null;
  entries: ArenaLeaderboardEntry[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function getArenaLeaderboard(signal?: AbortSignal): Promise<ArenaLeaderboard> {
  const res = await fetch(`${API_BASE}/v1/arena/leaderboard`, {
    headers: { Accept: "application/json" },
    signal,
  });
  if (!res.ok) throw new Error(`arena leaderboard -> ${res.status}`);
  return (await res.json()) as ArenaLeaderboard;
}

export function useArenaLeaderboardQuery() {
  return useQuery({
    queryKey: ["arena", "leaderboard"],
    queryFn: ({ signal }: { signal: AbortSignal }) => getArenaLeaderboard(signal),
  });
}
