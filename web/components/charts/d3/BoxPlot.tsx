// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import { BoxPlotProps } from "@/types/chart.types";
import { useThemeColors } from "@/hooks/useThemeColors";

// Match the text sizes on the timeline chart above: 14px axis labels, 12px
// tick/legend/category text.
const modelFontSize = 12;
const providerFontSize = 12;
const axisLabelFontSize = "14px";
const yAxisTickFontSize = "12px";
const modelLineHeight = 14;

const BoxPlot: React.FC<BoxPlotProps> = ({
  data,
  width = 800,
  height = 500,
  getModelColor,
  getProviderForModel,
  normalizeModelName,
  isMobile = false
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({
    width: width || 800,
    height
  });
  const themeColors = useThemeColors();

  // Handle responsive sizing — always fit the card's content box.
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current) return;
      setDimensions({
        width: containerRef.current.offsetWidth,
        height: height
      });
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    // Track container size changes that happen without a window resize
    const resizeObserver = new ResizeObserver(handleResize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      resizeObserver.disconnect();
    };
  }, [height]);

  useEffect(() => {
    if (!svgRef.current || data.data.length === 0) return;

    // STT hides the y-axis tick labels, so it needs almost no left margin —
    // reclaim that space for a wider plot. TTS keeps room for the "1.4s" ticks.
    const showYTicks = !(
      data.metricType === "NTTFT" ||
      data.metricType === "TTFT" ||
      data.metricType === "TTFS"
    );
    const margin = { top: 20, right: 8, bottom: 80, left: showYTicks ? 40 : 10 };
    const chartWidth = dimensions.width - margin.left - margin.right;
    const chartHeight = dimensions.height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Use the sorted order from the data (already sorted by median)
    const xScale = d3
      .scaleBand()
      .domain(data.data.map((d) => d.model))
      .range([0, chartWidth])
      .paddingInner(0.2)
      .paddingOuter(0);

    // Ticks every 0.5 s. The domain snaps to those boundaries so the
    // outermost gridlines — the visible floor and ceiling — enclose the
    // whiskers.
    const tickStepMs = 500;
    const yMin = Math.max(0, Math.floor(data.globalMin / tickStepMs) * tickStepMs);
    let yMax = Math.ceil(data.globalMax / tickStepMs) * tickStepMs;
    if (yMax === yMin) yMax = yMin + tickStepMs;
    const yScale = d3
      .scaleLinear()
      .domain([yMin, yMax])
      .range([chartHeight, 0]);

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Add Y axis
    const yAxis = d3
      .axisLeft(yScale)
      // d3 pixel-snaps ticks for crisp 1px lines; recharts doesn't. Disable
      // the snapping so the gridlines render as softly as the other charts'.
      .offset(0)
      .tickValues(d3.range(yMin, yMax + 1, tickStepMs))
      .tickSize(-chartWidth)
      // Seconds with up to two decimals, trailing zeros stripped, so
      // sub-second ticks stay distinct (0.25s, 0.3s) without clutter (1s, 2.2s).
      .tickFormat((d) => `${parseFloat((Number(d) / 1000).toFixed(2))}s`);

    const yAxisGroup = g.append("g")
      .attr("class", "y-axis")
      .call(yAxis);

    // Always show gridlines, dotted to match the recharts plots
    yAxisGroup.selectAll("line")
      .attr("stroke", themeColors.grid)
      .attr("stroke-dasharray", "2 2");

    // STT hides the y-axis tick labels (see showYTicks); TTS keeps them styled.
    if (showYTicks) {
      yAxisGroup.selectAll("text")
        .attr("fill", themeColors.axisText)
        .attr("font-size", yAxisTickFontSize);
    } else {
      yAxisGroup.selectAll("text").style("display", "none");
    }

    yAxisGroup.select(".domain").remove();

    // Add X axis
    const xAxis = d3.axisBottom(xScale);

    const xAxisGroup = g
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${chartHeight})`)
      .call(xAxis);

    xAxisGroup.selectAll("line").remove();
    xAxisGroup.select(".domain").remove();
    xAxisGroup.selectAll("text").remove();

    // Create custom wrapped text with model first (desktop only)
    if (!isMobile) {
      const labelMaxWidth = xScale.step() * 0.96;
      const minModelFont = 8;

      data.data.forEach((modelData) => {
        const model = modelData.model;
        const normalizedModel = normalizeModelName(model);
        const provider = getProviderForModel(model);

        const xPosition = (xScale(model) ?? 0) + xScale.bandwidth() / 2;
        const yPosition = chartHeight + 15; // Base position

        const maxCharsPerLine = 8;
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

        const modelTextNodes = modelLines.map((line, lineIndex) =>
          g
            .append("text")
            .attr("x", xPosition)
            .attr("y", yPosition + lineIndex * modelLineHeight)
            .attr("text-anchor", "middle")
            .attr("fill", themeColors.label)
            .attr("font-size", `${modelFontSize}px`)
            .attr("font-weight", "bold")
            .text(line)
        );

        const providerNode = g
          .append("text")
          .attr("x", xPosition)
          .attr("text-anchor", "middle")
          .attr("fill", themeColors.axisText)
          .attr("font-size", `${providerFontSize}px`)
          .text(provider);

        let widest = providerNode.node()?.getComputedTextLength() ?? 0;
        modelTextNodes.forEach((node) => {
          widest = Math.max(widest, node.node()?.getComputedTextLength() ?? 0);
        });

        const scale =
          widest > labelMaxWidth
            ? Math.max(minModelFont / modelFontSize, labelMaxWidth / widest)
            : 1;
        const lineHeight = modelLineHeight * scale;

        modelTextNodes.forEach((node, lineIndex) => {
          node
            .attr("font-size", `${modelFontSize * scale}px`)
            .attr("y", yPosition + lineIndex * lineHeight);
        });

        providerNode
          .attr("font-size", `${providerFontSize * scale}px`)
          .attr("y", yPosition + modelLines.length * lineHeight + 1);
      });
    }

    // X-axis label (the Y axis title is omitted — it's redundant with the
    // card heading).
    g.append("text")
      .attr(
        "transform",
        `translate(${chartWidth / 2}, ${chartHeight + margin.bottom - 10})`
      )
      .style("text-anchor", "middle")
      .attr("fill", themeColors.axisText)
      .attr("font-size", axisLabelFontSize)
      .text("Ranked by P50 latency");

    // Render a box plot for each model
    data.data.forEach((modelData) => {
      const color = getModelColor(modelData.model);
      const { min, q1, median, q3, max } = modelData.quartiles;

      const boxGroup = g
        .append("g")
        .attr(
          "class",
          `box-${modelData.model.replace(/[^a-zA-Z0-9]/g, "-")}`
        );

      const centerX = (xScale(modelData.model) ?? 0) + xScale.bandwidth() / 2;
      const boxWidth = Math.min(xScale.bandwidth() * 0.5, 48);

      // Main box (IQR)
      boxGroup
        .append("rect")
        .attr("x", centerX - boxWidth / 2)
        .attr("y", yScale(q3))
        .attr("width", boxWidth)
        .attr("height", yScale(q1) - yScale(q3))
        .attr("fill", color)
        .attr("fill-opacity", 0.3)
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("rx", 2);

      // Median line
      boxGroup
        .append("line")
        .attr("x1", centerX - boxWidth / 2)
        .attr("x2", centerX + boxWidth / 2)
        .attr("y1", yScale(median))
        .attr("y2", yScale(median))
        .attr("stroke", themeColors.median)
        .attr("stroke-opacity", 0.55)
        .attr("stroke-width", 1.5);

      // Whiskers
      const whiskerWidth = boxWidth * 0.8;

      // Lower whisker
      boxGroup
        .append("line")
        .attr("x1", centerX)
        .attr("x2", centerX)
        .attr("y1", yScale(q1))
        .attr("y2", yScale(min))
        .attr("stroke", color)
        .attr("stroke-width", 1.5);

      boxGroup
        .append("line")
        .attr("x1", centerX - whiskerWidth / 2)
        .attr("x2", centerX + whiskerWidth / 2)
        .attr("y1", yScale(min))
        .attr("y2", yScale(min))
        .attr("stroke", color)
        .attr("stroke-width", 1.5);

      // Upper whisker
      boxGroup
        .append("line")
        .attr("x1", centerX)
        .attr("x2", centerX)
        .attr("y1", yScale(q3))
        .attr("y2", yScale(max))
        .attr("stroke", color)
        .attr("stroke-width", 1.5);

      boxGroup
        .append("line")
        .attr("x1", centerX - whiskerWidth / 2)
        .attr("x2", centerX + whiskerWidth / 2)
        .attr("y1", yScale(max))
        .attr("y2", yScale(max))
        .attr("stroke", color)
        .attr("stroke-width", 1.5);

      // Add interactive hover effects
      boxGroup
        .style("cursor", "pointer")
        .on("mouseenter", function () {
          d3.select(this)
            .select("rect")
            .attr("fill-opacity", 0.5);

          const tooltip = g.append("g").attr("class", "box-tooltip");

          const tooltipRect = tooltip
            .append("rect")
            .attr("x", 0)
            .attr("y", -40)
            .attr("height", 80)
            .attr("fill", themeColors.tooltipBg)
            .attr("stroke", color)
            .attr("stroke-width", 1)
            .attr("rx", 6)
            .attr("opacity", 0.95);

          const tooltipLines = [
            {
              text: modelData.model,
              y: -25,
              size: "12px",
              weight: "bold",
              fill: themeColors.tooltipText,
            },
            {
              text: `Count: ${modelData.stats.count}`,
              y: -8,
              size: "10px",
              weight: "normal",
              fill: themeColors.tooltipSecondary,
            },
            {
              text: `Mean: ${modelData.stats.mean.toFixed(0)}ms`,
              y: 6,
              size: "10px",
              weight: "normal",
              fill: themeColors.tooltipSecondary,
            },
            {
              text: `Median: ${modelData.quartiles.median.toFixed(0)}ms`,
              y: 20,
              size: "10px",
              weight: "normal",
              fill: themeColors.tooltipSecondary,
            },
            {
              text: `Std: ${modelData.stats.std.toFixed(0)}ms`,
              y: 34,
              size: "10px",
              weight: "normal",
              fill: themeColors.tooltipSecondary,
            },
          ];

          const tooltipNodes = tooltipLines.map((line) => {
            const node = tooltip
              .append("text")
              .attr("x", 8)
              .attr("y", line.y)
              .attr("fill", line.fill)
              .attr("font-size", line.size)
              .attr("font-weight", line.weight)
              .text(line.text);
            return {
              node,
              width: node.node()?.getComputedTextLength() ?? 0,
              size: line.size,
            };
          });

          const tooltipTextWidth = tooltipNodes.reduce(
            (max, n) => Math.max(max, n.width),
            0
          );
          const tooltipWidth = Math.min(
            chartWidth,
            Math.max(140, tooltipTextWidth + 16)
          );
          tooltipRect.attr("width", tooltipWidth);

          const innerWidth = tooltipWidth - 16;
          tooltipNodes.forEach(({ node, width, size }) => {
            if (width > innerWidth && width > 0) {
              const baseSize = parseFloat(size);
              node.attr("font-size", `${baseSize * (innerWidth / width)}px`);
            }
          });

          const tooltipX = Math.max(
            0,
            Math.min(
              centerX + 30 + tooltipWidth > chartWidth
                ? centerX - 30 - tooltipWidth
                : centerX + 30,
              chartWidth - tooltipWidth
            )
          );
          tooltip.attr(
            "transform",
            `translate(${tooltipX}, ${yScale(median)})`
          );
        })
        .on("mouseleave", function () {
          d3.select(this)
            .select("rect")
            .attr("fill-opacity", 0.3);

          g.select(".box-tooltip").remove();
        });
    });
  }, [data, dimensions, getModelColor, getProviderForModel, normalizeModelName, isMobile, themeColors]);

  // The measured container must always render — an early return here would
  // leave the sizing effect's ResizeObserver attached to nothing, freezing
  // dimensions at their initial value once data arrives.
  return (
    <div ref={containerRef} className="w-full">
      {data.data.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-text-secondary">
          <p>No data available for box plot</p>
        </div>
      ) : (
        <svg
          ref={svgRef}
          width={dimensions.width}
          height={dimensions.height}
          className="overflow-visible"
          style={{ background: "transparent" }}
        />
      )}
    </div>
  );
};

export default BoxPlot;
