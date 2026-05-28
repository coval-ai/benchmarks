import { getEnabledSttModels, getEnabledTtsModels } from "@/lib/playground/providers";

// SECURITY LIMITATION: `Origin`, `x-forwarded-for`, and `x-real-ip` are all
// caller-controlled — any HTTP client can forge them. These checks are
// defense-in-depth for honest browser clients only. The in-process Maps below
// are not shared across Vercel instances (each gets fresh state on cold start).
// Upgrade path: signed session token + Upstash Redis before exposing to
// untrusted traffic at scale.
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

// Per-IP caps mirror runner/src/coval_bench/api/ratelimit.py (in-memory,
// single-instance). Upgrade path: Turnstile + Redis if abuse appears.
export const MAX_SESSIONS_PER_IP = Math.max(
  getEnabledTtsModels().length,
  getEnabledSttModels().length,
);
export const DAILY_REQUEST_CAP = MAX_SESSIONS_PER_IP * 5;

const concurrent = new Map<string, number>();

export function tryAcquireSession(ip: string): boolean {
  const cur = concurrent.get(ip) ?? 0;
  if (cur >= MAX_SESSIONS_PER_IP) return false;
  concurrent.set(ip, cur + 1);
  return true;
}

export function releaseSession(ip: string): void {
  const cur = concurrent.get(ip) ?? 0;
  if (cur <= 1) concurrent.delete(ip);
  else concurrent.set(ip, cur - 1);
}

const DAILY_TTL_MS = 24 * 60 * 60 * 1000;
const PRUNE_THRESHOLD = 10_000;
const daily = new Map<string, { count: number; resetAt: number }>();

export function tryConsumeDailyQuota(ip: string): boolean {
  if (daily.size > PRUNE_THRESHOLD) pruneExpired();
  const now = Date.now();
  const entry = daily.get(ip);
  if (!entry || now > entry.resetAt) {
    daily.set(ip, { count: 1, resetAt: now + DAILY_TTL_MS });
    return true;
  }
  if (entry.count >= DAILY_REQUEST_CAP) return false;
  entry.count++;
  return true;
}

function pruneExpired(): void {
  const now = Date.now();
  for (const [ip, entry] of daily) {
    if (now > entry.resetAt) daily.delete(ip);
  }
}
