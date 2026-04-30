/**
 * Build a URL query string from a params object.
 * Omits keys whose value is undefined or null.
 * All other values are serialized via URLSearchParams (coerced to string).
 */
export function buildQueryString(
  params: Record<string, string | number | boolean | null | undefined>
): string {
  const entries = Object.entries(params).filter(
    ([, v]) => v !== undefined && v !== null
  ) as [string, string | number | boolean][];

  if (entries.length === 0) return "";

  const qs = new URLSearchParams(
    entries.map(([k, v]) => [k, String(v)])
  );
  return `?${qs.toString()}`;
}
