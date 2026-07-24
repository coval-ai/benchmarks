// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { normalizeModelName } from "@/lib/utils/formatters";
import { useThemeColors } from "@/hooks/useThemeColors";

// Match the text sizes on the timeline chart: 12px category/legend text.
const labelFontPx = 12;
/** Longest label kept before it is ellipsized, per layout. */
const maxLabelChars = { compact: 12, mobile: 14, desktop: 18 };
/** Geist Mono advances a fixed 0.6em, so label widths are predictable. */
const monoAdvance = 0.6;
/**
 * Geist Mono advances 0.6em and the label hangs at 45°, so a label reaches this
 * far left of its own tick. The axis must reserve it as left padding or the
 * first label is clipped at the plot edge — it reads as cut off by the y-axis.
 * The model line is ellipsized to the budget above but the provider line is
 * not, so pass the providers on show: a longer one sets the reach.
 */
export const tickLabelReach = (isMobile: boolean, providers: string[] = []) =>
  Math.ceil(
    Math.max(
      isMobile ? maxLabelChars.mobile : maxLabelChars.desktop,
      ...providers.map((provider) => provider.length)
    ) *
      labelFontPx *
      monoAdvance *
      Math.SQRT1_2
  );

const CustomBarChartTick: React.FC<{
  x?: number;
  y?: number;
  width?: number;
  visibleTicksCount?: number;
  payload?: { value: string };
  getProviderForModel: (model: string) => string;
  isMobile?: boolean;
}> = ({
  x = 0,
  y = 0,
  width = 0,
  visibleTicksCount = 1,
  payload,
  getProviderForModel,
  isMobile = false,
}) => {
  const themeColors = useThemeColors();

  if (!payload) return null;

  const model = payload.value;
  const fullName = normalizeModelName(model);
  const provider = getProviderForModel(model);

  // Labels sit on a 45° diagonal, so adjacent labels are slot·sin(45°) apart
  // perpendicular to the text. Two 12px lines need ~40px of slot; below that
  // the provider line collides with the neighbour's model line, so drop it
  // and shorten the name (the tooltip and <title> still carry the full info).
  const slot = visibleTicksCount > 0 ? width / visibleTicksCount : width;
  const showProvider = slot >= 40;
  const maxChars = !showProvider
    ? maxLabelChars.compact
    : isMobile
      ? maxLabelChars.mobile
      : maxLabelChars.desktop;
  const normalizedModel =
    fullName.length > maxChars ? `${fullName.slice(0, maxChars - 1)}…` : fullName;
  const showTitle = !showProvider || normalizedModel !== fullName;

  // Render every label on a single 45° diagonal so they don't collide when many
  // models are shown: model name (bold) on the first line, provider below it,
  // both anchored at the tick so the text fans out down-left.
  return (
    <g transform={`translate(${x},${y})`}>
      <g transform="rotate(-45)">
        <text
          x={0}
          y={0}
          dy={14}
          textAnchor="end"
          fill={themeColors.label}
          fontSize={labelFontPx}
          fontWeight="bold"
        >
          {showTitle && <title>{`${provider} ${fullName}`}</title>}
          {normalizedModel}
        </text>
        {showProvider && (
          <text
            x={0}
            y={0}
            dy={28}
            textAnchor="end"
            fill={themeColors.axisText}
            fontSize={labelFontPx}
          >
            {provider}
          </text>
        )}
      </g>
    </g>
  );
};

export default CustomBarChartTick;
