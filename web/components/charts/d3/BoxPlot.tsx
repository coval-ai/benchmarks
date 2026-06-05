// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import { BoxPlotProps } from "@/types/chart.types";
import { useThemeColors } from "@/hooks/useThemeColors";

// Match the text sizes on the timeline chart above: 14px axis labels, 12px
// tick/legend/category text.
const modelFontSize = "12px";
const providerFontSize = "12px";
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
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: height
        });
      } else {
        // Fallback for initial render - use a more conservative width
        const viewportWidth = window.innerWidth;
        setDimensions({
          width: viewportWidth * 0.85, // Use 85% of viewport width
          height: height
        });
      }
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
      data.metricType === "NTTFT" || data.metricType === "TTFT"
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

    const yScale = d3
      .scaleLinear()
      .domain([Math.max(0, data.globalMin - 50), data.globalMax * 1.05])
      .range([chartHeight, 0]);

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Add Y axis
    const yAxis = d3
      .axisLeft(yScale)
      .tickSize(-chartWidth)
      // Seconds with up to two decimals, trailing zeros stripped, so
      // sub-second ticks stay distinct (0.25s, 0.3s) without clutter (1s, 2.2s).
      .tickFormat((d) => `${parseFloat((Number(d) / 1000).toFixed(2))}s`);

    const yAxisGroup = g.append("g")
      .attr("class", "y-axis")
      .call(yAxis);

    // Always show gridlines
    yAxisGroup.selectAll("line")
      .attr("stroke", themeColors.grid);

    yAxisGroup.selectAll("text")
      .attr("fill", themeColors.axisText)
      .attr("font-size", yAxisTickFontSize);

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
      data.data.forEach((modelData) => {
        const model = modelData.model;
        const normalizedModel = normalizeModelName(model);
        const provider = getProviderForModel(model);

        const xPosition = (xScale(model) ?? 0) + xScale.bandwidth() / 2;
        const yPosition = chartHeight + 15; // Base position

        // Wrap model name if too long (max ~15 characters per line)
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

        // Add model name lines with dynamic font size
        modelLines.forEach((line, lineIndex) => {
          g.append("text")
            .attr("x", xPosition)
            .attr("y", yPosition + lineIndex * modelLineHeight)
            .attr("text-anchor", "middle")
            .attr("fill", themeColors.label)
            .attr("font-size", modelFontSize)
            .attr("font-weight", "bold")
            .text(line);
        });

        // Add provider name with dynamic font size
        const providerYPosition =
          yPosition + modelLines.length * modelLineHeight + 1;
        g.append("text")
          .attr("x", xPosition)
          .attr("y", providerYPosition)
          .attr("text-anchor", "middle")
          .attr("fill", themeColors.axisText)
          .attr("font-size", providerFontSize)
          .text(provider);
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

          const tooltip = g
            .append("g")
            .attr("class", "box-tooltip")
            .attr("transform", `translate(${centerX + 30}, ${yScale(median)})`);

          tooltip
            .append("rect")
            .attr("x", 0)
            .attr("y", -40)
            .attr("width", 140)
            .attr("height", 80)
            .attr("fill", themeColors.tooltipBg)
            .attr("stroke", color)
            .attr("stroke-width", 1)
            .attr("rx", 6)
            .attr("opacity", 0.95);

          tooltip
            .append("text")
            .attr("x", 8)
            .attr("y", -25)
            .attr("fill", themeColors.tooltipText)
            .attr("font-size", "12px")
            .attr("font-weight", "bold")
            .text(modelData.model);

          tooltip
            .append("text")
            .attr("x", 8)
            .attr("y", -8)
            .attr("fill", themeColors.tooltipSecondary)
            .attr("font-size", "10px")
            .text(`Count: ${modelData.stats.count}`);

          tooltip
            .append("text")
            .attr("x", 8)
            .attr("y", 6)
            .attr("fill", themeColors.tooltipSecondary)
            .attr("font-size", "10px")
            .text(`Mean: ${modelData.stats.mean.toFixed(0)}ms`);

          tooltip
            .append("text")
            .attr("x", 8)
            .attr("y", 20)
            .attr("fill", themeColors.tooltipSecondary)
            .attr("font-size", "10px")
            .text(`Median: ${modelData.quartiles.median.toFixed(0)}ms`);

          tooltip
            .append("text")
            .attr("x", 8)
            .attr("y", 34)
            .attr("fill", themeColors.tooltipSecondary)
            .attr("font-size", "10px")
            .text(`Std: ${modelData.stats.std.toFixed(0)}ms`);
        })
        .on("mouseleave", function () {
          d3.select(this)
            .select("rect")
            .attr("fill-opacity", 0.3);

          g.select(".box-tooltip").remove();
        });
    });
  }, [data, dimensions, getModelColor, getProviderForModel, normalizeModelName, isMobile, themeColors]);

  if (data.data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-text-secondary">
        <p>No data available for box plot</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="overflow-visible"
        style={{ background: "transparent" }}
      />
    </div>
  );
};

export default BoxPlot;
