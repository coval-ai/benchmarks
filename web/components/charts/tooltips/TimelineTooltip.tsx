import React from "react";
import { formatTimeWithSeconds, normalizeModelName } from "@/lib/utils/formatters";

interface TimelineTooltipProps {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    name: string;
    color: string;
  }>;
  label?: string | number;
  getProviderForModel: (model: string) => string;
}

const CustomTimelineTooltip: React.FC<TimelineTooltipProps> = ({ active, payload, label, getProviderForModel }) => {
  if (!active || !payload || payload.length === 0) return null;

  // Filter out null/undefined values and sort by value (fastest to slowest)
  const validData = payload
    .filter((item) => item.value != null && item.value > 0)
    .sort((a, b) => a.value - b.value); // Ascending = fastest first

  if (validData.length === 0) return null;

  return (
    <div
      style={{
        backgroundColor: "var(--color-surface-tooltip)",
        border: "1px solid var(--color-border-secondary)",
        borderRadius: "8px",
        color: "var(--color-text-on-tooltip)",
        padding: "12px",
        minWidth: "250px"
      }}
    >
      <p
        style={{ margin: "0 0 8px 0", fontWeight: "bold", fontSize: "12px" }}
      >
        {formatTimeWithSeconds(Number(label))}
      </p>
      <div style={{ fontSize: "11px" }}>
        {validData.map((item, index) => {
          // Extract model name from dataKey (remove '_value' suffix)
          const modelName = item.dataKey.replace("_value", "");
          const provider = getProviderForModel(modelName);

          return (
            <div
              key={item.dataKey}
              style={{
                margin: "6px 0",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", flex: 1 }}>
                <div
                  style={{
                    width: "8px",
                    height: "8px",
                    backgroundColor: item.color,
                    borderRadius: "50%",
                    marginRight: "8px",
                    flexShrink: 0
                  }}
                />
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      color: index === 0 ? "#10B981" : "var(--color-text-on-tooltip)",
                      fontWeight: index === 0 ? "bold" : "normal"
                    }}
                  >
                    #{index + 1} {normalizeModelName(modelName)}
                  </div>
                  <div
                    style={{
                      color: "var(--color-text-on-tooltip-secondary)",
                      fontSize: "10px",
                      marginTop: "1px"
                    }}
                  >
                    {provider}
                  </div>
                </div>
              </div>
              <span
                style={{
                  color: "var(--color-text-on-tooltip-secondary)",
                  marginLeft: "12px",
                  fontWeight: index === 0 ? "bold" : "normal",
                  flexShrink: 0
                }}
              >
                {item.value.toFixed(0)}ms
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CustomTimelineTooltip;
