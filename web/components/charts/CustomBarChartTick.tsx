import React from "react";
import { useThemeColors } from "@/hooks/useThemeColors";

// Helper function to normalize model names
const normalizeModelName = (modelName: string): string => {
  // Mapping mirrors the backend's enabled provider matrix (runner/config.py).
  const modelMappings: Record<string, string> = {
    // TTS
    "tts-1-hd": "TTS 1 HD",
    eleven_multilingual_v2: "Multilingual v2",
    eleven_flash_v2_5: "Flash v2.5",
    eleven_turbo_v2_5: "Turbo v2.5",
    "sonic-3": "Sonic 3",
    "aura-2-thalia-en": "Aura 2",
    arcana: "Arcana",
    mistv3: "Mist v3",
    // STT
    "nova-2": "Nova 2",
    "nova-3": "Nova 3",
    "flux-general-en": "Flux",
    scribe_v2_realtime: "Scribe v2",
    "universal-streaming": "Universal Streaming",
    default: "Default",
    enhanced: "Enhanced"
  };

  // Return mapped name if it exists
  if (modelMappings[modelName]) {
    return modelMappings[modelName];
  }

  // Fallback: automatic normalization for unmapped models
  return modelName
    .replace(/-/g, " ") // Replace hyphens with spaces
    .split(" ") // Split into words
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1)) // Capitalize
    .join(" "); // Join back together
};

const CustomBarChartTick: React.FC<{
  x?: number;
  y?: number;
  payload?: { value: string };
  getProviderForModel: (model: string) => string;
  isMobile?: boolean;
  sidebarCollapsed?: boolean;
}> = ({ x = 0, y = 0, payload, getProviderForModel, isMobile = false, sidebarCollapsed = true }) => {
  const themeColors = useThemeColors();

  if (!payload) return null;

  const model = payload.value;
  const normalizedModel = normalizeModelName(model);
  const provider = getProviderForModel(model);

  // Adjust font sizes based on sidebar state
  const modelFontSize = sidebarCollapsed ? "12px" : "10px";
  const providerFontSize = sidebarCollapsed ? "10px" : "9px";
  const mobileFontSize = sidebarCollapsed ? "11px" : "10px";

  // Mobile: Show only model name, diagonal
  if (isMobile) {
    return (
      <g transform={`translate(${x},${y})`}>
        <text
          x={0}
          y={0}
          dy={16}
          textAnchor="end"
          fill={themeColors.label}
          fontSize={mobileFontSize}
          fontWeight="bold"
          transform="rotate(-45)"
        >
          {normalizedModel}
        </text>
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
