import { Redis } from "@upstash/redis";
import { getEnabledSttModels, getEnabledTtsModels } from "./providers";

const ALLOWED_ORIGINS = new Set([
  "http://localhost:3000",
  "https://benchmarks.coval.ai",
]);

export function isAllowedOrigin(origin: string | null): boolean {
  if (!origin) return false;
  return ALLOWED_ORIGINS.has(origin);
}

export const MAX_CONCURRENT_PER_SESSION = Math.max(
  getEnabledTtsModels().length,
  getEnabledSttModels().length,
);

export const STT_DAILY_CAP = MAX_CONCURRENT_PER_SESSION * 5;
export const TTS_DAILY_CAP = MAX_CONCURRENT_PER_SESSION * 5;

export type QuotaKind = "stt" | "tts";

const CONCURRENCY_TTL_S = 120;
const DAILY_TTL_S = 24 * 60 * 60;

function resolveRedisCreds(): { url: string; token: string } | null {
  const prefix = process.env.PLAYGROUND_REDIS_ENV_PREFIX?.replace(/_+$/, "");
  const prefixed = (suffix: string) =>
    prefix ? process.env[`${prefix}_${suffix}`] : undefined;
  const url =
    process.env.UPSTASH_REDIS_REST_URL ||
    process.env.KV_REST_API_URL ||
    prefixed("KV_REST_API_URL") ||
    "";
  const token =
    process.env.UPSTASH_REDIS_REST_TOKEN ||
    process.env.KV_REST_API_TOKEN ||
    prefixed("KV_REST_API_TOKEN") ||
    "";
  return url && token ? { url, token } : null;
}

const creds = resolveRedisCreds();
const redis = creds ? new Redis(creds) : null;

const ACQUIRE_LUA = `
local n = redis.call('INCR', KEYS[1])
if n > tonumber(ARGV[1]) then
  redis.call('DECR', KEYS[1])
  return 0
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
return 1`;

const RELEASE_LUA = `
local n = redis.call('DECR', KEYS[1])
if n <= 0 then redis.call('DEL', KEYS[1]) end
return n`;

const QUOTA_LUA = `
local n = redis.call('INCRBY', KEYS[1], tonumber(ARGV[1]))
if n == tonumber(ARGV[1]) then
  redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
end
if n > tonumber(ARGV[2]) then
  local after = redis.call('DECRBY', KEYS[1], tonumber(ARGV[1]))
  if after <= 0 then redis.call('DEL', KEYS[1]) end
  return 0
end
return 1`;

export async function tryAcquireSession(sid: string): Promise<boolean> {
  if (!redis) return tryAcquireSessionMemory(sid);
  try {
    const res = await redis.eval(
      ACQUIRE_LUA,
      [`pg:conc:${sid}`],
      [MAX_CONCURRENT_PER_SESSION, CONCURRENCY_TTL_S],
    );
    return Number(res) === 1;
  } catch (err) {
    console.error("[playground/security] acquire failed, denying", err);
    return false;
  }
}

export async function releaseSession(sid: string): Promise<void> {
  if (!redis) return releaseSessionMemory(sid);
  try {
    await redis.eval(RELEASE_LUA, [`pg:conc:${sid}`], []);
  } catch (err) {
    console.error("[playground/security] release failed", err);
  }
}

export async function tryConsumeDailyQuota(
  sid: string,
  kind: QuotaKind,
  count: number = 1,
): Promise<boolean> {
  if (count <= 0) return true;
  const cap = kind === "stt" ? STT_DAILY_CAP : TTS_DAILY_CAP;
  if (!redis) return tryConsumeDailyQuotaMemory(sid, kind, count, cap);
  try {
    const res = await redis.eval(
      QUOTA_LUA,
      [`pg:daily:${kind}:${sid}`],
      [count, cap, DAILY_TTL_S],
    );
    return Number(res) === 1;
  } catch (err) {
    console.error("[playground/security] quota check failed, denying", err);
    return false;
  }
}

const DAILY_TTL_MS = DAILY_TTL_S * 1000;
const CONCURRENCY_TTL_MS = CONCURRENCY_TTL_S * 1000;
const PRUNE_THRESHOLD = 10_000;
const concurrent = new Map<string, { count: number; resetAt: number }>();
const dailyStt = new Map<string, { count: number; resetAt: number }>();
const dailyTts = new Map<string, { count: number; resetAt: number }>();

function tryAcquireSessionMemory(sid: string): boolean {
  if (concurrent.size > PRUNE_THRESHOLD) pruneExpired(concurrent);
  const now = Date.now();
  let entry = concurrent.get(sid);
  if (!entry || now > entry.resetAt) entry = { count: 0, resetAt: now + CONCURRENCY_TTL_MS };
  if (entry.count >= MAX_CONCURRENT_PER_SESSION) return false;
  entry.count += 1;
  entry.resetAt = now + CONCURRENCY_TTL_MS;
  concurrent.set(sid, entry);
  return true;
}

function releaseSessionMemory(sid: string): void {
  const entry = concurrent.get(sid);
  if (!entry) return;
  if (entry.count <= 1) concurrent.delete(sid);
  else entry.count -= 1;
}

function tryConsumeDailyQuotaMemory(
  sid: string,
  kind: QuotaKind,
  count: number,
  cap: number,
): boolean {
  const map = kind === "stt" ? dailyStt : dailyTts;
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
