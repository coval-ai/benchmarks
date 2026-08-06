// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import type { PricingEntry } from "@/lib/api/client";
import { formatUsd, normalizeModelName, toModelKey } from "@/lib/utils/formatters";
import { billingUnitLabel, priceUnitShortLabel } from "@/lib/utils/pricing";

// Provenance URLs are provider-controlled data by way of the collector;
// only http(s) may become a navigable link (defense-in-depth against a
// malformed or malicious scheme ever reaching the table).
function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

/**
 * The one provenance body every displayed price carries (AA convention):
 * native rate(s), derivation basis, plan assumption, as-of date, source link.
 * The link is reachable once the host tooltip is pinned (tap/click).
 */
export function priceTooltipContent(
  entry: PricingEntry,
  tab: "tts" | "stt" | "s2s"
): React.ReactNode {
  const conversion = entry.conversion;
  const measured = entry.basis === "list_price_measured_conversion";
  return (
    <>
      <span className="font-semibold">
        {normalizeModelName(toModelKey(entry.provider, entry.model))}:{" "}
        {entry.normalized_usd != null
          ? `${formatUsd(entry.normalized_usd)} ${priceUnitShortLabel(tab)}`
          : "no normalized price"}
      </span>
      {entry.native_rates.map((rate) => (
        <span key={rate.billing_unit} className="block opacity-80">
          List rate: {formatUsd(rate.rate_usd)} {billingUnitLabel(rate.billing_unit)}
        </span>
      ))}
      {measured && (
        <span className="block opacity-80">
          * Converted with rates measured from our runs
          {conversion
            ? ` (${[
                conversion.in_tokens_per_min != null &&
                  `${Math.round(conversion.in_tokens_per_min)} in-tokens/min`,
                conversion.out_tokens_per_min != null &&
                  `${Math.round(conversion.out_tokens_per_min)} out-tokens/min`,
                conversion.chars_per_sec != null &&
                  `${conversion.chars_per_sec.toFixed(1)} chars/s`,
              ]
                .filter(Boolean)
                .join(", ")}, ${conversion.window} window)`
            : ""}
        </span>
      )}
      {entry.normalized_usd == null && (
        <span className="block opacity-80">
          Normalizing needs usage measured from our runs; not enough samples yet.
        </span>
      )}
      {entry.native_rates.some((r) => r.plan_assumption) && (
        <span className="block opacity-80">
          Plan assumption: {entry.native_rates.find((r) => r.plan_assumption)?.plan_assumption}
        </span>
      )}
      <span className="block pt-1 opacity-80">
        as of {entry.as_of} ·{" "}
        {isHttpUrl(entry.source_url) ? (
          <a
            href={entry.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2"
            onClick={(e) => e.stopPropagation()}
          >
            provider pricing page
          </a>
        ) : (
          <span>{entry.source_url}</span>
        )}
      </span>
    </>
  );
}
