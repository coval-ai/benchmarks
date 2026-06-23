const BASE = process.env.ARENA_API_URL ?? "";
const KEY = process.env.ARENA_LABELER_KEY ?? "";

export async function arenaRunnerFetch(path: string, body?: unknown): Promise<Response> {
  if (!BASE) throw new Error("ARENA_API_URL is not configured");
  const headers = new Headers({ "X-Labeler-Key": KEY });
  if (body === undefined) {
    return fetch(`${BASE}${path}`, { method: "GET", headers });
  }
  headers.set("Content-Type", "application/json");
  return fetch(`${BASE}${path}`, { method: "POST", headers, body: JSON.stringify(body) });
}
