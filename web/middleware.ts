import { NextResponse, type NextRequest } from "next/server";
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

export function middleware(req: NextRequest) {
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

export const config = {
  matcher: "/playground",
};
