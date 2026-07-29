import { NextResponse, type NextRequest } from "next/server";
import {
  ACCESS_COOKIE_MAX_AGE_S,
  ACCESS_COOKIE_NAME,
  ACCESS_COOKIE_PATHS,
  type ArenaRole,
  accessSecretConfigured,
  gateAllows,
  identified,
  legacyLabeler,
  mintAccess,
  needsRefresh,
  safeEqual,
  verifyAccess,
} from "@/lib/arena/access";
import { COOKIE_NAME, COOKIE_PATHS, mintSession, verifySession } from "@/lib/playground/session";

export const runtime = "nodejs";

function buildCookieHeader(name: string, value: string, path: string, expires: Date): string {
  const parts = [
    `${name}=${value}`,
    `Path=${path}`,
    `Expires=${expires.toUTCString()}`,
    "HttpOnly",
    "SameSite=Lax",
  ];
  if (process.env.NODE_ENV === "production") parts.push("Secure");
  return parts.join("; ");
}

/**
 * Issue the arena identity cookie over both scoped paths, retiring any unscoped one.
 *
 * The pre-scoping cookie sat at "/" and still prefix-matches /arena, so leaving it in
 * place would have the browser send two cookies of the same name and pick between them
 * by path length. Expiring it in the same response keeps that from happening, and costs
 * the visitor nothing.
 */
function setIdentityCookie(res: NextResponse, token: string): void {
  const expires = new Date(Date.now() + ACCESS_COOKIE_MAX_AGE_S * 1000);
  for (const path of ACCESS_COOKIE_PATHS) {
    res.headers.append("Set-Cookie", buildCookieHeader(ACCESS_COOKIE_NAME, token, path, expires));
  }
  res.headers.append(
    "Set-Cookie",
    buildCookieHeader(ACCESS_COOKIE_NAME, "", "/", new Date(0)),
  );
}

/** The unlock link, when it carries the right token: mint a labeler and clean the URL. */
function unlockedAsLabeler(req: NextRequest): NextResponse | null {
  const token = process.env.ARENA_ACCESS_TOKEN;
  const provided = req.nextUrl.searchParams.get("access");
  if (!token || !provided || !safeEqual(provided, token)) return null;
  const url = req.nextUrl.clone();
  url.searchParams.delete("access");
  const res = NextResponse.redirect(url);
  setIdentityCookie(res, mintAccess("labeler"));
  return res;
}

function reissue(res: NextResponse, role: ArenaRole, sid?: string): void {
  setIdentityCookie(res, sid === undefined ? mintAccess(role) : mintAccess(role, sid));
}

/**
 * Say so when identity is off, rather than serving an arena that quietly attributes
 * nothing.
 *
 * Rate-limited rather than once-per-instance: a warm instance would otherwise report the
 * outage once and then go quiet for as long as it lives, which is the wrong shape for
 * something you want to alert on.
 */
const IDENTITY_REPORT_INTERVAL_MS = 10 * 60 * 1000;
let identityLastReportedAt = 0;

function reportIdentityDisabled(): void {
  if (process.env.NODE_ENV !== "production") return;
  const now = Date.now();
  if (identityLastReportedAt !== 0 && now - identityLastReportedAt < IDENTITY_REPORT_INTERVAL_MS) {
    return;
  }
  identityLastReportedAt = now;
  console.error(
    "arena identity disabled: ARENA_SESSION_SECRET is missing or shorter than the required " +
      "32 characters, so no visitor is issued a signed identity and no vote can be attributed",
  );
}

/**
 * Give every visitor to a public arena surface a signed identity.
 *
 * Votes are attributed to the `sid` in this cookie, so a visitor without one cannot be
 * held to a quota or judged by later scoring. Cookies predating identity belong to the
 * internal labelers who were here before the arena opened, so they are reissued as
 * `labeler`; everyone arriving since is `external`.
 */
function ensureArenaIdentity(req: NextRequest): NextResponse {
  // Serve the arena anyway — a misconfigured secret should not take the public page
  // down — but never silently.
  if (!accessSecretConfigured()) {
    reportIdentityDisabled();
    return NextResponse.next();
  }

  const unlocked = unlockedAsLabeler(req);
  if (unlocked) return unlocked;

  const res = NextResponse.next();
  const payload = verifyAccess(req.cookies.get(ACCESS_COOKIE_NAME)?.value);
  const current = payload && identified(payload);
  if (!payload) reissue(res, "external");
  else if (legacyLabeler(payload)) reissue(res, "labeler");
  else if (!current) reissue(res, "external");
  else if (needsRefresh(current)) reissue(res, current.role, current.sid);
  return res;
}

function arenaGate(req: NextRequest): NextResponse {
  // Local dev always sees the arena; only prod/preview require the token.
  if (process.env.NODE_ENV !== "production") return NextResponse.next();

  const token = process.env.ARENA_ACCESS_TOKEN;
  if (!token || !accessSecretConfigured()) return new NextResponse(null, { status: 404 });

  const unlocked = unlockedAsLabeler(req);
  if (unlocked) return unlocked;

  // A signature alone is not enough: every public visitor now holds a validly signed
  // `external` cookie, and it is sent here because it is scoped to /arena. Only a labeler
  // gets through.
  const payload = verifyAccess(req.cookies.get(ACCESS_COOKIE_NAME)?.value);
  if (payload && gateAllows(payload)) {
    const res = NextResponse.next();
    const current = identified(payload);
    if (!current) reissue(res, "labeler");
    else if (needsRefresh(current)) reissue(res, current.role, current.sid);
    return res;
  }

  // Locked: indistinguishable from a route that does not exist.
  return new NextResponse(null, { status: 404 });
}

function playgroundSession(req: NextRequest): NextResponse {
  const response = NextResponse.next();
  const existing = verifySession(req.cookies.get(COOKIE_NAME)?.value);
  if (existing) return response;

  const { token, expiresAt } = mintSession();
  const expires = new Date(expiresAt);
  for (const path of COOKIE_PATHS) {
    response.headers.append("Set-Cookie", buildCookieHeader(COOKIE_NAME, token, path, expires));
  }
  return response;
}

// Exactly two arena surfaces are public — the voting page and the leaderboard — plus
// the endpoints they call. The arena needs traffic to tighten its provisional
// confidence intervals, and the runner caps every arena endpoint at 60/minute.
//
// This is an allowlist on purpose: anything else under /arena or /api/arena stays
// behind the access token, so a route added later is gated by default rather than
// published by omission. /arena/admin and /api/arena/admin/* are the current such
// routes — convergence and Elo pairing internals, with no auth of their own.
const PUBLIC_ARENA_PAGES = new Set(["/arena", "/arena/leaderboard"]);
const PUBLIC_ARENA_API_PREFIXES = [
  "/api/arena/battle", // create a battle, and /battle/<id>/reveal after the vote
  "/api/arena/vote",
  "/api/arena/example-prompt",
  "/api/arena/leaderboard",
];

function isPublicArenaPath(pathname: string): boolean {
  if (PUBLIC_ARENA_PAGES.has(pathname)) return true;
  return PUBLIC_ARENA_API_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  );
}

export function middleware(req: NextRequest): NextResponse {
  const { pathname } = req.nextUrl;
  if (pathname.startsWith("/arena") || pathname.startsWith("/api/arena")) {
    return isPublicArenaPath(pathname) ? ensureArenaIdentity(req) : arenaGate(req);
  }
  return playgroundSession(req);
}

export const config = {
  matcher: ["/playground", "/arena", "/arena/:path*", "/api/arena/:path*"],
};
