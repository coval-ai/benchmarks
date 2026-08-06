// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useMemo, useRef } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import { useDashboard } from "@/contexts/DashboardContext";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { getModelColor } from "@/lib/utils/colors";
import { formatDate, formatUsd, normalizeModelName, parseModelKey } from "@/lib/utils/formatters";
import { priceUnitShortLabel } from "@/lib/utils/pricing";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

interface HistorySeries {
  model: string;
  provider: string;
  /** Real recorded points, oldest first. */
  points: { t: number; price: number }[];
  /** points + a synthetic "still effective now" tail for the step line. */
  plotted: { t: number; price: number }[];
}

// Prices come from the append-only pricing table, so a model's history is its
// full life of list rates; a step line is the honest shape (a rate holds until
// superseded, it never drifts).
const PriceHistorySection: React.FC = () => {
  const { selectedModels, pricingByModel, getProviderForModel } = useDashboard();
  const activeTab = useActiveTab();
  const themeColors = useThemeColors();
  const trackChartHover = useChartHoverTracking("price_history");

  const series = useMemo<HistorySeries[]>(() => {
    const now = Date.now();
    return selectedModels.flatMap((model) => {
      const entry = pricingByModel.get(model);
      if (!entry) return [];
      const points = entry.history
        .filter((h) => h.normalized_usd != null)
        .map((h) => ({ t: Date.parse(h.effective_at), price: h.normalized_usd! }))
        .filter((p) => Number.isFinite(p.t))
        .sort((a, b) => a.t - b.t);
      const last = points[points.length - 1];
      if (!last) return [];
      return [
        {
          model,
          provider: getProviderForModel(model),
          points,
          plotted: [...points, { t: now, price: last.price }],
        },
      ];
    });
  }, [selectedModels, pricingByModel, getProviderForModel]);

  const changed = useMemo(() => series.filter((s) => s.points.length >= 2), [series]);
  const hasHistory = changed.length > 0;

  const viewTracked = useRef(false);
  useEffect(() => {
    if (!hasHistory || viewTracked.current) return;
    viewTracked.current = true;
    capturePostHogEvent(POSTHOG_EVENTS.priceHistoryViewed, {
      surface: `${activeTab}_dashboard`,
      mode: activeTab,
      models_with_changes: changed.length,
    });
  }, [hasHistory, changed.length, activeTab]);

  // No pricing feature at all on this board — no card either.
  if (pricingByModel.size === 0) return null;

  const unit = priceUnitShortLabel(activeTab);

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8" onMouseEnter={trackChartHover}>
        <SectionHeader
          label="Price History"
          description={{
            short: `List price over time (${unit})`,
            detailed:
              "Every published rate change we record, as a step line: a price holds until the provider supersedes it. Prices are normalized list rates with full provenance in the comparison table's price column.",
          }}
          exportXLabel="Effective date"
          exportRows={() =>
            changed.flatMap(({ model, provider, points }) =>
              points.map((p) => ({
                model: parseModelKey(model).model,
                provider,
                [activeTab === "tts" ? "price_usd_per_1m_chars" : "price_usd_per_1k_min"]:
                  p.price,
                effective_at: new Date(p.t).toISOString(),
              }))
            )
          }
          exportImage={hasHistory}
        />
        {hasHistory ? (
          <div data-export-frame className="h-64">
            <ResponsiveContainer width="100%" height="100%" debounce={200}>
              <LineChart margin={{ top: 10, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={themeColors.grid} strokeDasharray="2 2" />
                <XAxis
                  dataKey="t"
                  type="number"
                  scale="time"
                  domain={["dataMin", "dataMax"]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: themeColors.axisText, fontSize: 12 }}
                  tickFormatter={(t) => formatDate(Number(t))}
                />
                <YAxis
                  width={56}
                  domain={[0, "auto"]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: themeColors.axisText, fontSize: 12 }}
                  tickFormatter={(v) => formatUsd(Number(v))}
                />
                <Tooltip
                  isAnimationActive={false}
                  labelFormatter={(t) => formatDate(Number(t))}
                  formatter={(value, name) => [
                    `${formatUsd(Number(value))} ${unit}`,
                    normalizeModelName(String(name)),
                  ]}
                  contentStyle={{
                    backgroundColor: "var(--color-surface-tooltip)",
                    border: "1px solid var(--color-border-secondary)",
                    borderRadius: "8px",
                    color: "var(--color-text-on-tooltip)",
                  }}
                />
                {changed.map(({ model, plotted }) => (
                  <Line
                    key={model}
                    data={plotted}
                    dataKey="price"
                    name={model}
                    type="stepAfter"
                    stroke={getModelColor(model)}
                    strokeWidth={2}
                    dot={{ r: 3, fill: getModelColor(model) }}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="flex h-24 items-center justify-center text-sm text-text-tertiary">
            No price changes recorded yet — every selected model has held its
            current list rate.
          </p>
        )}
      </Card>
    </div>
  );
};

export default PriceHistorySection;
