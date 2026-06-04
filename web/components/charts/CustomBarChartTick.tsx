// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { normalizeModelName } from "@/lib/utils/formatters";
import { useThemeColors } from "@/hooks/useThemeColors";

const modelFontSize = "10px";
const providerFontSize = "9px";
const mobileFontSize = "10px";

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

  // Mobile: model name + provider, diagonal
  if (isMobile) {
    return (
      <g transform={`translate(${x},${y})`}>
        <g transform="rotate(-45)">
          <text
            x={0}
            y={0}
            dy={14}
            textAnchor="end"
            fill={themeColors.label}
            fontSize={mobileFontSize}
            fontWeight="bold"
          >
            {normalizedModel}
          </text>
          <text
            x={0}
            y={0}
            dy={26}
            textAnchor="end"
            fill={themeColors.label}
            fontSize={providerFontSize}
            opacity={0.8}
          >
            {provider}
          </text>
        </g>
      </g>
    );
  }

  // Desktop: Show wrapped model name + provider
  const maxCharsPerLine = 10;
  const modelWords = normalizedModel.split(/[-_\s]/);
  const modelLines: string[] = [];
  let currentLine = "";

  modelWords.forEach((word) => {
    if ((currentLine + word).length <= maxCharsPerLine) {
      currentLine += (currentLine ? "-" : "") + word;
    } else {
      if (currentLine) {
        modelLines.push(currentLine);
        currentLine = word;
      } else {
        modelLines.push(word);
      }
    }
  });

  if (currentLine) {
    modelLines.push(currentLine);
  }

  return (
    <g transform={`translate(${x},${y})`}>
      {/* Model name lines (wrapped) */}
      {modelLines.map((line, lineIndex) => {
        const dy = 16 + lineIndex * 16;
        return (
          <text
            key={line}
            x={0}
            y={0}
            dy={dy}
            textAnchor="middle"
            fill={themeColors.label}
            fontSize={modelFontSize}
            fontWeight="bold"
          >
            {line}
          </text>
        );
      })}

      {/* Provider name (below model) */}
      <text
        x={0}
        y={0}
        dy={16 + modelLines.length * 16 + 12}
        textAnchor="middle"
        fill={themeColors.axisText}
        fontSize={providerFontSize}
      >
        {provider}
      </text>
    </g>
  );
};

export default CustomBarChartTick;
