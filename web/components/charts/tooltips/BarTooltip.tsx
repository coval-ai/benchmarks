import React from "react";
import { CustomBarTooltipProps } from "@/types/chart.types";

const CustomBarTooltip: React.FC<CustomBarTooltipProps> = ({
  active,
  payload,
  label
}) => {
  if (active && payload && payload.length > 0) {
    const value = payload[0]?.value;
    return (
      <div
        style={{
          backgroundColor: "var(--color-surface-tooltip)",
          border: "1px solid var(--color-border-secondary)",
          borderRadius: "8px",
          padding: "8px 12px"
        }}
      >
        <p
          style={{ margin: 0, fontWeight: "bold", color: "var(--color-text-on-tooltip)" }}
        >{`Model: ${label}`}</p>
        <p style={{ margin: 0, color: "var(--color-text-on-tooltip)" }}>{`Average WER: ${Number(
          value
        ).toFixed(1)}%`}</p>
      </div>
    );
  }
  return null;
};

export default CustomBarTooltip;
