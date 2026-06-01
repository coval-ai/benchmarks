import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";

const SECRET_ENV = "PLAYGROUND_SESSION_SECRET";
const SECRET_PREVIOUS_ENV = "PLAYGROUND_SESSION_SECRET_PREVIOUS";
const TTL_MS = 24 * 60 * 60 * 1000;

export const COOKIE_NAME = "__playground_session";
export const COOKIE_PATHS = ["/playground", "/api/playground"] as const;

export type SessionPayload = { sid: string; iat: number; exp: number };

function requireSecret(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not configured`);
  return value;
}

function activeSecrets(): { primary: string; previous: string | null } {
  return { primary: requireSecret(SECRET_ENV), previous: process.env[SECRET_PREVIOUS_ENV] ?? null };
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

export function mintSession(): { token: string; expiresAt: number; sid: string } {
  const { primary } = activeSecrets();
  const sid = randomUUID();
  const iat = Date.now();
  const exp = iat + TTL_MS;
  const payload: SessionPayload = { sid, iat, exp };
  const body = b64url(Buffer.from(JSON.stringify(payload)));
  const signature = sign(body, primary);
  return { token: `${body}.${signature}`, expiresAt: exp, sid };
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  return timingSafeEqual(Buffer.from(a), Buffer.from(b));
}

export function verifySession(token: string | null | undefined): SessionPayload | null {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 2) return null;
  const [body, sig] = parts as [string, string];

  const { primary, previous } = activeSecrets();
  const candidates = previous ? [primary, previous] : [primary];
  const matches = candidates.some((secret) => constantTimeEqual(sig, sign(body, secret)));
  if (!matches) return null;

  let payload: SessionPayload;
  try {
    payload = JSON.parse(b64urlDecode(body).toString("utf8")) as SessionPayload;
  } catch {
    return null;
  }
  if (typeof payload?.sid !== "string" || payload.sid.length === 0) return null;
  if (typeof payload.exp !== "number" || Date.now() > payload.exp) return null;
  return payload;
}

function readCookieFromHeader(cookieHeader: string | null, name: string): string | null {
  if (!cookieHeader) return null;
  for (const part of cookieHeader.split(/;\s*/)) {
    const eq = part.indexOf("=");
    if (eq > 0 && part.slice(0, eq) === name) return part.slice(eq + 1);
  }
  return null;
}

export function getSessionFromRequest(req: Request): SessionPayload | null {
  return verifySession(readCookieFromHeader(req.headers.get("cookie"), COOKIE_NAME));
}
