import { createHmac, timingSafeEqual } from "node:crypto";

const SECRET_ENV = "ARENA_SESSION_SECRET";
const TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days of inactivity before lock-out
const REFRESH_AFTER_MS = 24 * 60 * 60 * 1000; // re-issue at most once/day (sliding window)

export const ACCESS_COOKIE_NAME = "arena_access";
export const ACCESS_COOKIE_MAX_AGE_S = TTL_MS / 1000;

export type AccessPayload = { iat: number; exp: number };

function requireSecret(): string {
  const value = process.env[SECRET_ENV];
  if (!value) throw new Error(`${SECRET_ENV} is not configured`);
  return value;
}

function b64url(buf: Buffer): string {
  return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s: string): Buffer {
  const pad = "=".repeat((4 - (s.length % 4)) % 4);
  return Buffer.from(s.replace(/-/g, "+").replace(/_/g, "/") + pad, "base64");
}

function sign(body: string, secret: string): string {
  return b64url(createHmac("sha256", secret).update(body).digest());
}

export function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  return timingSafeEqual(Buffer.from(a), Buffer.from(b));
}

export function mintAccess(now: number = Date.now()): string {
  const payload: AccessPayload = { iat: now, exp: now + TTL_MS };
  const body = b64url(Buffer.from(JSON.stringify(payload)));
  return `${body}.${sign(body, requireSecret())}`;
}

export function verifyAccess(
  token: string | null | undefined,
  now: number = Date.now(),
): AccessPayload | null {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 2) return null;
  const [body, sig] = parts as [string, string];
  if (!safeEqual(sig, sign(body, requireSecret()))) return null;

  let payload: AccessPayload;
  try {
    payload = JSON.parse(b64urlDecode(body).toString("utf8")) as AccessPayload;
  } catch {
    return null;
  }
  if (typeof payload?.iat !== "number" || typeof payload?.exp !== "number") return null;
  if (now > payload.exp) return null;
  return payload;
}

export function needsRefresh(payload: AccessPayload, now: number = Date.now()): boolean {
  return now - payload.iat > REFRESH_AFTER_MS;
}
