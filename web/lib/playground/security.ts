import { getEnabledSttModels, getEnabledTtsModels } from "@/lib/playground/providers";

// Defense-in-depth pre-check. The real abuse boundary is the signed session
// cookie verified in `web/lib/playground/session.ts`; `Origin` and forwarded-IP
// headers are still caller-controlled and used only as cheap early rejection
// and for diagnostic logging. The in-process Maps below are not shared across
// Vercel instances (each gets fresh state on cold start). Upgrade path:
// Upstash Redis keyed on session id (Phase 2) before scaling beyond a single
// Vercel instance reliably.
const ALLOWED_ORIGINS = new Set([
  "http://localhost:3000",
  "https://benchmarks.coval.ai",
  // Add specific Vercel preview URLs here when testing playground on a preview deploy.
]);

export function isAllowedOrigin(origin: string | null): boolean {
  if (!origin) return false;
  return ALLOWED_ORIGINS.has(origin);
}

export function getClientIp(req: Request): string {
  const fwd = req.headers.get("x-forwarded-for");
  if (fwd) {
    const first = fwd.split(",")[0];
    if (first) return normalizeIp(first.trim());
  }
  const real = req.headers.get("x-real-ip");
  if (real) return normalizeIp(real);
  return "unknown";
}

function normalizeIp(ip: string): string {
  return ip.replace(/^::ffff:/, "");
}

export const MAX_CONCURRENT_PER_SESSION = Math.max(
  getEnabledTtsModels().length,
  getEnabledSttModels().length,
);

export const STT_DAILY_CAP = MAX_CONCURRENT_PER_SESSION * 5;
export const TTS_DAILY_CAP = MAX_CONCURRENT_PER_SESSION * 5;

const concurrent = new Map<string, number>();

export function tryAcquireSession(sid: string): boolean {
  const cur = concurrent.get(sid) ?? 0;
  if (cur >= MAX_CONCURRENT_PER_SESSION) return false;
  concurrent.set(sid, cur + 1);
  return true;
}

export function releaseSession(sid: string): void {
  const cur = concurrent.get(sid) ?? 0;
  if (cur <= 1) concurrent.delete(sid);
  else concurrent.set(sid, cur - 1);
}

const DAILY_TTL_MS = 24 * 60 * 60 * 1000;
const PRUNE_THRESHOLD = 10_000;
const dailyStt = new Map<string, { count: number; resetAt: number }>();
const dailyTts = new Map<string, { count: number; resetAt: number }>();

export type QuotaKind = "stt" | "tts";

export function tryConsumeDailyQuota(sid: string, kind: QuotaKind, count: number = 1): boolean {
  if (count <= 0) return true;
  const map = kind === "stt" ? dailyStt : dailyTts;
  const cap = kind === "stt" ? STT_DAILY_CAP : TTS_DAILY_CAP;
  if (map.size > PRUNE_THRESHOLD) pruneExpired(map);
  const now = Date.now();
  let entry = map.get(sid);
  if (!entry || now > entry.resetAt) {
    entry = { count: 0, resetAt: now + DAILY_TTL_MS };
    map.set(sid, entry);
  }
  if (entry.count + count > cap) return false;
  entry.count += count;
  return true;
}

function pruneExpired(map: Map<string, { count: number; resetAt: number }>): void {
  const now = Date.now();
  for (const [sid, entry] of map) {
    if (now > entry.resetAt) map.delete(sid);
  }
}
