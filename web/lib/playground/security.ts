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
export const DAILY_REQUEST_CAP = MAX_CONCURRENT_PER_SESSION * 5;

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
const daily = new Map<string, { count: number; resetAt: number }>();

export function tryConsumeDailyQuota(sid: string): boolean {
  if (daily.size > PRUNE_THRESHOLD) pruneExpired();
  const now = Date.now();
  const entry = daily.get(sid);
  if (!entry || now > entry.resetAt) {
    daily.set(sid, { count: 1, resetAt: now + DAILY_TTL_MS });
    return true;
  }
  if (entry.count >= DAILY_REQUEST_CAP) return false;
  entry.count++;
  return true;
}

function pruneExpired(): void {
  const now = Date.now();
  for (const [sid, entry] of daily) {
    if (now > entry.resetAt) daily.delete(sid);
  }
}
