import { NextResponse, type NextRequest } from "next/server";
import {
  ACCESS_COOKIE_MAX_AGE_S,
  ACCESS_COOKIE_NAME,
  mintAccess,
  needsRefresh,
  safeEqual,
  verifyAccess,
} from "@/lib/arena/access";
import { COOKIE_NAME, COOKIE_PATHS, mintSession, verifySession } from "@/lib/playground/session";

export const runtime = "nodejs";

function buildCookieHeader(value: string, path: string, expires: Date): string {
  const parts = [
    `${COOKIE_NAME}=${value}`,
    `Path=${path}`,
    `Expires=${expires.toUTCString()}`,
    "HttpOnly",
    "SameSite=Lax",
  ];
  if (process.env.NODE_ENV === "production") parts.push("Secure");
  return parts.join("; ");
}

function setAccessCookie(res: NextResponse): void {
  res.cookies.set(ACCESS_COOKIE_NAME, mintAccess(), {
    httpOnly: true,
    sameSite: "lax",
    secure: true,
    path: "/",
    maxAge: ACCESS_COOKIE_MAX_AGE_S,
  });
}

function arenaGate(req: NextRequest): NextResponse {
  // Local dev always sees the arena; only prod/preview require the token.
  if (process.env.NODE_ENV !== "production") return NextResponse.next();

  const token = process.env.ARENA_ACCESS_TOKEN;
  const secret = process.env.ARENA_SESSION_SECRET;
  if (!token || !secret) return new NextResponse(null, { status: 404 }); // fail closed if misconfigured

  // Unlock link: ?access=<token> -> mint cookie, redirect to the clean URL.
  const provided = req.nextUrl.searchParams.get("access");
  if (provided && safeEqual(provided, token)) {
    const url = req.nextUrl.clone();
    url.searchParams.delete("access");
    const res = NextResponse.redirect(url);
    setAccessCookie(res);
    return res;
  }

  const payload = verifyAccess(req.cookies.get(ACCESS_COOKIE_NAME)?.value);
  if (payload) {
    const res = NextResponse.next();
    if (needsRefresh(payload)) setAccessCookie(res); // sliding window
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
    response.headers.append("Set-Cookie", buildCookieHeader(token, path, expires));
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
    return isPublicArenaPath(pathname) ? NextResponse.next() : arenaGate(req);
  }
  return playgroundSession(req);
}

export const config = {
  matcher: ["/playground", "/arena", "/arena/:path*", "/api/arena/:path*"],
};
