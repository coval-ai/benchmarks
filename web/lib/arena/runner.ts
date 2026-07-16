const BASE = process.env.ARENA_API_URL ?? "";
const KEY = process.env.ARENA_LABELER_KEY ?? "";

export async function arenaRunnerFetch(
  path: string,
  body?: unknown,
  clientKey?: string | null,
): Promise<Response> {
  if (!BASE) throw new Error("ARENA_API_URL is not configured");
  if (!KEY) throw new Error("ARENA_LABELER_KEY is not configured");
  const headers = new Headers({ "X-Labeler-Key": KEY });
  if (clientKey) headers.set("X-Arena-Client", clientKey);
  if (body === undefined) {
    return fetch(`${BASE}${path}`, { method: "GET", headers });
  }
  headers.set("Content-Type", "application/json");
  return fetch(`${BASE}${path}`, { method: "POST", headers, body: JSON.stringify(body) });
}
