// Values must match ArenaDomain in runner/src/coval_bench/api/schemas.py — they are
// stored on battles and used as leaderboard keys ("all" is reserved for the aggregate board).
export const ARENA_DOMAINS = [
  { value: "customer-service", label: "Customer Service" },
  { value: "healthcare", label: "Healthcare" },
  { value: "sales", label: "Sales" },
  { value: "receptionist-booking", label: "Receptionist / Booking" },
  { value: "other", label: "Other" },
] as const;

export type ArenaDomain = (typeof ARENA_DOMAINS)[number]["value"];

export function isArenaDomain(value: unknown): value is ArenaDomain {
  return ARENA_DOMAINS.some((d) => d.value === value);
}
