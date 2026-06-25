import { cookies } from "next/headers";
import { ACCESS_COOKIE_NAME, verifyAccess } from "./access";

export async function arenaAccessOk(): Promise<boolean> {
  if (process.env.NODE_ENV !== "production") return true;
  const token = (await cookies()).get(ACCESS_COOKIE_NAME)?.value;
  return verifyAccess(token) !== null;
}
