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
  payload?: { value: string };
  getProviderForModel: (model: string) => string;
  isMobile?: boolean;
}> = ({ x = 0, y = 0, payload, getProviderForModel, isMobile = false }) => {
  const themeColors = useThemeColors();

  if (!payload) return null;

  const model = payload.value;
  const normalizedModel = normalizeModelName(model);
  const provider = getProviderForModel(model);

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
          {normalizedModel}
        </text>
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
      </g>
    </g>
  );
};

export default CustomBarChartTick;
