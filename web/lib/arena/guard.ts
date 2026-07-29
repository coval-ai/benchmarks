import { cookies } from "next/headers";
import { ACCESS_COOKIE_NAME, accessSecretConfigured, gateAllows, verifyAccess } from "./access";

/**
 * Whether the caller may read an admin route. Middleware gates these paths first; this is
 * the same decision made again at the route, so a matcher change cannot quietly open them.
 */
export async function arenaAccessOk(): Promise<boolean> {
  if (process.env.NODE_ENV !== "production") return true;
  if (!accessSecretConfigured()) return false;
  const payload = verifyAccess((await cookies()).get(ACCESS_COOKIE_NAME)?.value);
  return payload !== null && gateAllows(payload);
}
