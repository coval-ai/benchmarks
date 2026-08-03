import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";

const SECRET_ENV = "ARENA_SESSION_SECRET";
const SECRET_PREVIOUS_ENV = "ARENA_SESSION_SECRET_PREVIOUS";
const TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days of inactivity before lock-out
const REFRESH_AFTER_MS = 24 * 60 * 60 * 1000; // re-issue at most once/day (sliding window)

export const ACCESS_COOKIE_NAME = "arena_access";
export const ACCESS_COOKIE_MAX_AGE_S = TTL_MS / 1000;

// Disjoint paths, so a request matches exactly one and the browser never holds two
// cookies of this name. Cookies issued at the retired paths ("/" before scoping,
// "/arena" before the pages moved under /overview) are expired on sight.
export const ACCESS_COOKIE_PATHS = ["/overview/arena", "/api/arena"] as const;
export const RETIRED_ACCESS_COOKIE_PATHS = ["/", "/arena"] as const;

export type ArenaRole = "labeler" | "external";

/** A verified cookie. `sid`/`role` are absent on cookies minted before identity existed. */
export type AccessPayload = { sid?: string; role?: ArenaRole; iat: number; exp: number };

/** A cookie carrying identity — what the BFF requires before attributing a vote. */
export type IdentifiedAccess = { sid: string; role: ArenaRole; iat: number; exp: number };

function requireSecret(): string {
  const value = process.env[SECRET_ENV];
  if (!value) throw new Error(`${SECRET_ENV} is not configured`);
  return value;
}

// Below this, treat the secret as absent rather than sign with it: a short secret yields a
// forgeable cookie, and failing closed beats quietly accepting one. The previous secret is
// deliberately exempt — it is verification-only, and rejecting it would lock out the live
// cookies rotation exists to preserve.
const SECRET_MIN_LENGTH = 32;

export function accessSecretConfigured(): boolean {
  return (process.env[SECRET_ENV] ?? "").length >= SECRET_MIN_LENGTH;
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
  const aBuf = Buffer.from(a);
  const bBuf = Buffer.from(b);
  if (aBuf.length !== bBuf.length) return false;
  return timingSafeEqual(aBuf, bBuf);
}

/**
 * Issue a cookie for `role`, reusing `sid` when refreshing an existing one.
 *
 * Passing the previous `sid` is what keeps a voter recognisable across the sliding
 * refresh — minting a fresh one every 24h would weaken vote dedupe and strip any
 * cross-day signal from later scoring.
 */
export function mintAccess(
  role: ArenaRole,
  sid: string = randomUUID(),
  now: number = Date.now(),
): string {
  const payload: IdentifiedAccess = { sid, role, iat: now, exp: now + TTL_MS };
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

  const previous = process.env[SECRET_PREVIOUS_ENV];
  const secrets = previous ? [requireSecret(), previous] : [requireSecret()];
  if (!secrets.some((secret) => safeEqual(sig, sign(body, secret)))) return null;

  let payload: AccessPayload;
  try {
    payload = JSON.parse(b64urlDecode(body).toString("utf8")) as AccessPayload;
  } catch {
    return null;
  }
  if (typeof payload?.iat !== "number" || typeof payload?.exp !== "number") return null;
  if (now >= payload.exp) return null;
  // Identity is both fields or neither. Nothing mints a half-filled payload, and one that
  // arrives fails `identified` and so reads as a cookie predating identity — which the gate
  // honours as a labeler. An explicit `role: "external"` would then arrive as a grant.
  const hasSid = payload.sid !== undefined;
  const hasRole = payload.role !== undefined;
  if (hasSid !== hasRole) return null;
  if (hasSid && (typeof payload.sid !== "string" || payload.sid === "")) return null;
  if (hasRole && payload.role !== "labeler" && payload.role !== "external") return null;
  return payload;
}

/** Narrow a verified cookie to one that actually carries identity. */
export function identified(payload: AccessPayload): IdentifiedAccess | null {
  if (!payload.sid || !payload.role) return null;
  return { sid: payload.sid, role: payload.role, iat: payload.iat, exp: payload.exp };
}

/**
 * When a cookie predating identity stops counting as a labeler.
 *
 * Such cookies could only have come from the unlock link, so honouring them keeps the
 * internal labelers signed in across this change. None can be minted any more, and each
 * is reissued with a `sid` on its owner's next visit, so the allowance is spent well
 * before this date. The bound is wall-clock on purpose: `iat` sits inside the payload a
 * forger would control, so comparing against it would guard nothing.
 */
export const LEGACY_LABELER_UNTIL = Date.UTC(2026, 8, 1);

export function legacyLabeler(payload: AccessPayload, now: number = Date.now()): boolean {
  // Both fields absent, not merely `identified` returning null — that is also true of a
  // half-filled payload, which is a different thing and must not be honoured.
  return payload.sid === undefined && payload.role === undefined && now < LEGACY_LABELER_UNTIL;
}

/**
 * Whether a verified cookie may reach the gated arena surfaces.
 *
 * `external` is minted for anyone who merely loaded a public arena page, and its cookie is
 * path-scoped to /overview/arena and /api/arena — so it is sent to the gated routes too,
 * and the role is the only thing keeping it out of them.
 */
export function gateAllows(payload: AccessPayload, now: number = Date.now()): boolean {
  return identified(payload)?.role === "labeler" || legacyLabeler(payload, now);
}

export function needsRefresh(payload: AccessPayload, now: number = Date.now()): boolean {
  return now - payload.iat > REFRESH_AFTER_MS;
}
