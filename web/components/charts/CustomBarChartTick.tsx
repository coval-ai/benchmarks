// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { normalizeModelName } from "@/lib/utils/formatters";
import { useThemeColors } from "@/hooks/useThemeColors";

// Match the text sizes on the timeline chart: 12px category/legend text.
const modelFontSize = "12px";
const providerFontSize = "12px";
const mobileFontSize = "12px";

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
  const maxChars = !showProvider ? 12 : isMobile ? 14 : 18;
  const normalizedModel =
    fullName.length > maxChars ? `${fullName.slice(0, maxChars - 1)}…` : fullName;
  const showTitle = !showProvider || normalizedModel !== fullName;

  // Render every label on a single 45° diagonal so they don't collide when many
  // models are shown: model name (bold) on the first line, provider below it,
  // both anchored at the tick so the text fans out down-left.
  const fontSize = isMobile ? mobileFontSize : modelFontSize;

  return (
    <g transform={`translate(${x},${y})`}>
      <g transform="rotate(-45)">
        <text
          x={0}
          y={0}
          dy={14}
          textAnchor="end"
          fill={themeColors.label}
          fontSize={fontSize}
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
            fontSize={providerFontSize}
          >
            {provider}
          </text>
        )}
      </g>
    </g>
  );
};

export default CustomBarChartTick;
