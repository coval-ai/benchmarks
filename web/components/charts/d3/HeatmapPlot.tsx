"use client";

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { HeatmapProps, SortConfig } from "@/types/chart.types";
import { ModelHeatmapData } from "@/types/benchmark.types";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useThemeColors } from "@/hooks/useThemeColors";

const HeatmapPlot: React.FC<HeatmapProps> = ({
  data,
  width = 800,
  height = 400,
  formatChartLabel,
  getProviderForModel,
  isMobile = false
}) => {
  const activeTab = useActiveTab();
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width, height });
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: null,
    direction: "asc"
  });
  const themeColors = useThemeColors();

  // Define metrics based on active tab
  const metrics = useMemo(() => {
    return activeTab === "tts"
      ? [
          {
            key: "latencyP25" as keyof ModelHeatmapData,
            label: "P25 Latency",
            unit: "ms",
            lowerIsBetter: true
          },
          {
            key: "latencyP50" as keyof ModelHeatmapData,
            label: "P50 Latency",
            unit: "ms",
            lowerIsBetter: true
          },
          {
            key: "latencyP75" as keyof ModelHeatmapData,
            label: "P75 Latency",
            unit: "ms",
            lowerIsBetter: true
          },
          {
            key: "latencyIQR" as keyof ModelHeatmapData,
            label: "Latency IQR",
            unit: "ms",
            lowerIsBetter: true
          },
          {
            key: "avgWER" as keyof ModelHeatmapData,
            label: "Avg WER",
            unit: "%",
            lowerIsBetter: true
          },
          {
            key: "werStdDev" as keyof ModelHeatmapData,
            label: "WER Std Dev",
            unit: "%",
            lowerIsBetter: true
          }
        ]
      : [
          {
            key: "latencyP25" as keyof ModelHeatmapData,
            label: "P25 Delta",
            unit: "ms",
            lowerIsBetter: true
          },
          {
            key: "latencyP50" as keyof ModelHeatmapData,
            label: "P50 Delta",
            unit: "ms",
            lowerIsBetter: true
          },
          {
            key: "latencyP75" as keyof ModelHeatmapData,
            label: "P75 Delta",
            unit: "ms",
            lowerIsBetter: true
          },
          {
            key: "latencyIQR" as keyof ModelHeatmapData,
            label: "Delta IQR",
            unit: "ms",
            lowerIsBetter: true
          },
          {
            key: "avgWER" as keyof ModelHeatmapData,
            label: "Avg WER",
            unit: "%",
            lowerIsBetter: true
          },
          {
            key: "werStdDev" as keyof ModelHeatmapData,
            label: "WER Std Dev",
            unit: "%",
            lowerIsBetter: true
          }
        ];
  }, [activeTab]);

  // Calculate dynamic height based on data
  const cellHeight = 50;
  const headerHeight = 80;
  const calculatedHeight = headerHeight + data.length * cellHeight;

  // Handle responsive sizing
  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const containerWidth = entry.contentRect.width;
        setDimensions({
          width: Math.max(600, containerWidth - 32),
          height: calculatedHeight
        });
      }
    });

    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [calculatedHeight]);

  // Sort data based on current sort configuration
  const sortedData = React.useMemo(() => {
    if (!sortConfig.key) return data;

    return [...data].sort((a, b) => {
      const aVal = a[sortConfig.key!];
      const bVal = b[sortConfig.key!];

      if (aVal < bVal) return sortConfig.direction === "asc" ? -1 : 1;
      if (aVal > bVal) return sortConfig.direction === "asc" ? 1 : -1;
      return 0;
    });
  }, [data, sortConfig]);

  // Handle column header clicks
  const handleSort = (key: keyof ModelHeatmapData) => {
    setSortConfig((prev) => ({
      key,
      direction: prev.key === key && prev.direction === "asc" ? "desc" : "asc"
    }));
  };

  // Create color scales for each metric
  const getColorScale = useCallback(
    (metricKey: keyof ModelHeatmapData, lowerIsBetter: boolean) => {
      if (!data || !Array.isArray(data) || data.length === 0) {
        return () => themeColors.grid;
      }

      const values = data
        .map((d) =>
          d && typeof d[metricKey] === "number" ? Number(d[metricKey]) : 0
        )
        .filter((v) => !isNaN(v) && v >= 0);

      if (values.length === 0) return () => themeColors.grid;

      const min = Math.min(...values);
      const max = Math.max(...values);

      return (value: number) => {
        const normalized = max === min ? 0 : (value - min) / (max - min);

        // Calculate intensity based on whether lower or higher is better
        let intensity;
        if (lowerIsBetter) {
          intensity = 1 - normalized;
        } else {
          intensity = normalized;
        }

        let red, green, blue;

        if (intensity >= 0.5) {
          const t = (intensity - 0.5) / 0.5;
          red = Math.round(255 - (255 - 46) * t);
          green = Math.round(133 + (204 - 133) * t);
          blue = Math.round(27 + (64 - 27) * t);
        } else {
          const t = intensity / 0.5;
          red = Math.round(255);
          green = Math.round(20 + (133 - 20) * t);
          blue = Math.round(147 - (147 - 27) * t);
        }

        return `rgba(${red}, ${green}, ${blue}, 0.6)`;
      };
    },
    [data, themeColors.grid]
  );

  // Format values for display
  const formatValue = useCallback((value: number, unit: string, metricKey: keyof ModelHeatmapData) => {
    if (value === 0 && (metricKey.includes('latency') || metricKey.includes('Latency') || metricKey.includes('Delta'))) {
      return '0';
    }

    if (metricKey.includes('latency') || metricKey.includes('Latency') || metricKey.includes('Delta')) {
      return Math.round(value).toString();
    }
    if (unit === "%") {
      return value.toFixed(1);
    }
    if (unit === "x") {
      return value.toFixed(2);
    }
    return value.toFixed(2);
  }, []);

  useEffect(() => {
    if (!svgRef.current || sortedData.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = {
      top: 60,
      right: 30,
      bottom: 20,
      left: isMobile ? 100 : 180
    };
    const chartWidth = dimensions.width - margin.left - margin.right;
    const chartHeight = dimensions.height - margin.top - margin.bottom;

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const yScale = d3
      .scaleBand()
      .domain(sortedData.map((d) => d.model))
      .range([0, chartHeight])
      .padding(0);

    const xScale = d3
      .scaleBand()
      .domain(metrics.map((m) => m.key as string))
      .range([0, chartWidth])
      .padding(0);

    // Add column headers
    metrics.forEach((metric) => {
      const headerGroup = g
        .append("g")
        .attr("class", "column-header")
        .style("cursor", "pointer")
        .on("click", () => handleSort(metric.key));

      // Header background (invisible but clickable)
      headerGroup
        .append("rect")
        .attr("x", xScale(metric.key as string) ?? 0)
        .attr("y", -50)
        .attr("width", xScale.bandwidth())
        .attr("height", 40)
        .attr("fill", "transparent");

      // Header text
      const headerText = metric.label;
      const xPosition =
        (xScale(metric.key as string) ?? 0) + xScale.bandwidth() / 2;

      if (isMobile && headerText.includes(" ")) {
        const words = headerText.split(" ");
        words.forEach((word, index) => {
          headerGroup
            .append("text")
            .attr("x", xPosition)
            .attr("y", -35 + index * 12)
            .attr("text-anchor", "middle")
            .attr("fill", themeColors.label)
            .attr("font-size", "10px")
            .attr("font-weight", "bold")
            .text(word);
        });
      } else {
        headerGroup
          .append("text")
          .attr("x", xPosition)
          .attr("y", -25)
          .attr("text-anchor", "middle")
          .attr("fill", themeColors.label)
          .attr("font-size", "12px")
          .attr("font-weight", "bold")
          .text(headerText);
      }

      // Sort indicator
      if (sortConfig.key === metric.key) {
        headerGroup
          .append("text")
          .attr(
            "x",
            (xScale(metric.key as string) ?? 0) + xScale.bandwidth() / 2
          )
          .attr("y", -10)
          .attr("text-anchor", "middle")
          .attr("fill", themeColors.label)
          .attr("font-size", "10px")
          .text(sortConfig.direction === "asc" ? "\u2191" : "\u2193");
      }
    });

    // Draw heatmap cells
    sortedData.forEach((model) => {
      metrics.forEach((metric) => {
        const value = Number(model[metric.key]);
        const colorScale = getColorScale(metric.key, metric.lowerIsBetter);
        const color = colorScale(value);

        const cellGroup = g.append("g").attr("class", "heatmap-cell");

        // Cell background
        cellGroup
          .append("rect")
          .attr("x", xScale(metric.key as string) ?? 0)
          .attr("y", yScale(model.model) ?? 0)
          .attr("width", xScale.bandwidth())
          .attr("height", yScale.bandwidth())
          .attr("fill", color)
          .attr("stroke", "none")
          .attr("stroke-width", 0)
          .style("opacity", 1);

        // Cell text
        cellGroup
          .append("text")
          .attr(
            "x",
            (xScale(metric.key as string) ?? 0) + xScale.bandwidth() / 2
          )
          .attr("y", (yScale(model.model) ?? 0) + yScale.bandwidth() / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("fill", themeColors.label)
          .attr("font-size", Math.min(14, xScale.bandwidth() / 6))
          .attr("font-weight", "500")
          .text(formatValue(value, metric.unit, metric.key));
      });
    });

    // Add row labels (model names)
    sortedData.forEach((model) => {
      const provider = getProviderForModel(model.model);
      const yPosition = (yScale(model.model) ?? 0) + yScale.bandwidth() / 2;

      if (isMobile) {
        // Mobile: Provider on top, model below
        g.append("text")
          .attr("x", -10)
          .attr("y", yPosition - 8)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("fill", themeColors.axisText)
          .attr("font-size", "9px")
          .attr("font-weight", "500")
          .text(provider);

        g.append("text")
          .attr("x", -10)
          .attr("y", yPosition + 8)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("fill", themeColors.label)
          .attr("font-size", "10px")
          .text(model.model);
      } else {
        // Desktop: Single line as before
        const labelText = formatChartLabel(model.model, provider);
        g.append("text")
          .attr("x", -10)
          .attr("y", yPosition)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("fill", themeColors.label)
          .attr("font-size", "11px")
          .text(labelText);
      }
    });
  }, [
    sortedData,
    dimensions,
    metrics,
    sortConfig,
    activeTab,
    getColorScale,
    formatValue,
    formatChartLabel,
    getProviderForModel,
    isMobile,
    themeColors
  ]);

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-text-secondary">
        <p>No data available for heatmap</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="overflow-visible heatmap-container"
        style={{ background: "transparent" }}
      />
    </div>
  );
};

export default HeatmapPlot;
