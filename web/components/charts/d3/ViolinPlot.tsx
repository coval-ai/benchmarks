// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import { ViolinPlotProps } from "@/types/chart.types";
import { useThemeColors } from "@/hooks/useThemeColors";

const ViolinPlot: React.FC<ViolinPlotProps> = ({
  data,
  width = 800,
  height = 500,
  getModelColor,
  getProviderForModel,
  normalizeModelName,
  isMobile = false,
  sidebarCollapsed = true
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({
    width: width || 800,
    height
  });
  const themeColors = useThemeColors();

  // Define font sizes based on sidebar state
  const modelFontSize = sidebarCollapsed ? "12px" : "10px";
  const providerFontSize = sidebarCollapsed ? "10px" : "9px";
  const axisLabelFontSize = sidebarCollapsed ? "14px" : "12px";
  const yAxisTickFontSize = sidebarCollapsed ? "12px" : "10px";

  // Handle responsive sizing
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.offsetWidth;
        setDimensions({
          width: Math.max(400, containerWidth - 32),
          height: height
        });
      } else {
        // Fallback for initial render - use a more conservative width
        const viewportWidth = window.innerWidth;
        const estimatedAvailableWidth = viewportWidth * 0.85; // Use 85% of viewport width
        setDimensions({
          width: Math.max(400, estimatedAvailableWidth),
          height: height
        });
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    // Also listen for sidebar changes by checking container size periodically
    const resizeObserver = new ResizeObserver(handleResize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      resizeObserver.disconnect();
    };
  }, [height]);

  // Helper function to create violin path from density data
  const createViolinPath = (
    densityData: { value: number; density: number }[],
    xScale: d3.ScaleBand<string>,
    yScale: d3.ScaleLinear<number, number>,
    modelName: string,
    maxDensity: number
  ): string => {
    if (densityData.length === 0) return "";

    const bandwidth = xScale.bandwidth();
    const centerX = (xScale(modelName) ?? 0) + bandwidth / 2;
    const maxWidth = bandwidth * 0.4; // Max width of violin

    // Create path points for the right side of the violin
    const rightSidePoints: [number, number][] = densityData.map((point) => {
      const w = (point.density / maxDensity) * maxWidth;
      return [centerX + w, yScale(point.value)];
    });

    // Create path points for the left side (mirror of right side)
    const leftSidePoints: [number, number][] = densityData
      .slice()
      .reverse()
      .map((point) => {
        const w = (point.density / maxDensity) * maxWidth;
        return [centerX - w, yScale(point.value)];
      });

    // Combine both sides to create a closed path
    const allPoints = rightSidePoints.concat(leftSidePoints);

    // Create the path using D3's line generator with curve
    const line = d3
      .line()
      .x((d) => d[0])
      .y((d) => d[1])
      .curve(d3.curveBasis); // Smooth curves for organic violin shape

    return line(allPoints) ?? "";
  };

  useEffect(() => {
    if (!svgRef.current || data.data.length === 0) return;

    const margin = { top: 20, right: 30, bottom: 80, left: 60 };
    const chartWidth = dimensions.width - margin.left - margin.right;
    const chartHeight = dimensions.height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Use the sorted order from the data (already sorted by median)
    const xScale = d3
      .scaleBand()
      .domain(data.data.map((d) => d.model))
      .range([0, chartWidth])
      .padding(0.2);

    const yScale = d3
      .scaleLinear()
      .domain([Math.max(0, data.globalMin - 50), data.globalMax * 1.05])
      .range([chartHeight, 0]);

    // Find the maximum density across all models for consistent scaling
    const maxDensity = Math.max(
      ...data.data.flatMap((modelData) =>
        modelData.density.map((point) => point.density)
      )
    );

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Add Y axis
    const yAxis = d3
      .axisLeft(yScale)
      .tickSize(-chartWidth)
      .tickFormat((d) => `${(Number(d) / 1000).toFixed(1)}s`);

    const yAxisGroup = g.append("g")
      .attr("class", "y-axis")
      .call(yAxis);

    // Always show gridlines
    yAxisGroup.selectAll("line")
      .attr("stroke", themeColors.grid)
      .attr("stroke-dasharray", "2,2");

    // Hide text labels for STT, show for TTS with dynamic font size
    if (data.metricType === "NTTFT" || data.metricType === "TTFT") {
      // STT - hide text labels only
      yAxisGroup.selectAll("text").style("display", "none");
    } else {
      // TTS - show text labels with dynamic styling
      yAxisGroup.selectAll("text")
        .attr("fill", themeColors.axisText)
        .attr("font-size", yAxisTickFontSize);
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
            .attr("y", yPosition + lineIndex * 12)
            .attr("text-anchor", "middle")
            .attr("fill", themeColors.label)
            .attr("font-size", modelFontSize)
            .attr("font-weight", "bold")
            .text(line);
        });

        // Add provider name with dynamic font size
        const providerYPosition = yPosition + modelLines.length * 12 + 8;
        g.append("text")
          .attr("x", xPosition)
          .attr("y", providerYPosition)
          .attr("text-anchor", "middle")
          .attr("fill", themeColors.axisText)
          .attr("font-size", providerFontSize)
          .text(provider);
      });
    }

    // Add axis labels with dynamic font size
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - chartHeight / 2)
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .attr("fill", themeColors.axisText)
      .attr("font-size", axisLabelFontSize)
      .text(`${data.metricType} (ms)`);

    g.append("text")
      .attr(
        "transform",
        `translate(${chartWidth / 2}, ${chartHeight + margin.bottom - 10})`
      )
      .style("text-anchor", "middle")
      .attr("fill", themeColors.axisText)
      .attr("font-size", axisLabelFontSize)
      .text("Ranked by P50 latency");

    // Render violin plots for each model
    data.data.forEach((modelData) => {
      const color = getModelColor(modelData.model);
      const { min, q1, median, q3, max, outliers } = modelData.quartiles;

      // Create a group for this model's violin plot
      const violinGroup = g
        .append("g")
        .attr(
          "class",
          `violin-${modelData.model.replace(/[^a-zA-Z0-9]/g, "-")}`
        );

      // 1. Render the violin shape (density curve)
      if (modelData.density.length > 0) {
        const visibleDensity = modelData.density.filter(
          (point) => point.value <= data.globalMax
        );

        const violinPath = createViolinPath(
          visibleDensity,
          xScale,
          yScale,
          modelData.model,
          maxDensity
        );

        violinGroup
          .append("path")
          .attr("d", violinPath)
          .attr("fill", color)
          .attr("fill-opacity", 0.3)
          .attr("stroke", color)
          .attr("stroke-width", 1)
          .attr("stroke-opacity", 0.8);
      }

      // 2. Overlay box plot elements on top of violin
      const centerX = (xScale(modelData.model) ?? 0) + xScale.bandwidth() / 2;
      const boxWidth = Math.min(xScale.bandwidth() * 0.15, 20);

      const boxGroup = violinGroup
        .append("g")
        .attr("class", "box-plot-overlay");

      // Main box (IQR)
      boxGroup
        .append("rect")
        .attr("x", centerX - boxWidth / 2)
        .attr("y", yScale(q3))
        .attr("width", boxWidth)
        .attr("height", yScale(q1) - yScale(q3))
        .attr("fill", themeColors.boxFill)
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
        .attr("stroke-width", 3);

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

      // Upper whisker - cap at visible maximum
      const visibleMax = Math.min(max, data.globalMax);

      boxGroup
        .append("line")
        .attr("x1", centerX)
        .attr("x2", centerX)
        .attr("y1", yScale(q3))
        .attr("y2", yScale(visibleMax))
        .attr("stroke", color)
        .attr("stroke-width", 1.5);

      boxGroup
        .append("line")
        .attr("x1", centerX - whiskerWidth / 2)
        .attr("x2", centerX + whiskerWidth / 2)
        .attr("y1", yScale(visibleMax))
        .attr("y2", yScale(visibleMax))
        .attr("stroke", color)
        .attr("stroke-width", 1.5);

      // Outliers - only plot those within the visible range
      const visibleOutliers = outliers.filter(
        (outlier) => outlier <= data.globalMax
      );
      visibleOutliers.forEach((outlier) => {
        boxGroup
          .append("circle")
          .attr("cx", centerX)
          .attr("cy", yScale(outlier))
          .attr("r", 4)
          .attr("fill", color)
          .attr("fill-opacity", 0.8)
          .attr("stroke", themeColors.median)
          .attr("stroke-width", 1);
      });

      // Add interactive hover effects
      violinGroup
        .style("cursor", "pointer")
        .on("mouseenter", function () {
          d3.select(this)
            .select("path")
            .attr("fill-opacity", 0.5)
            .attr("stroke-width", 2);

          const tooltip = g
            .append("g")
            .attr("class", "violin-tooltip")
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
            .select("path")
            .attr("fill-opacity", 0.3)
            .attr("stroke-width", 1);

          g.select(".violin-tooltip").remove();
        });
    });
  }, [data, dimensions, getModelColor, getProviderForModel, normalizeModelName, isMobile, sidebarCollapsed, modelFontSize, providerFontSize, axisLabelFontSize, yAxisTickFontSize, themeColors]);

  if (data.data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-text-secondary">
        <p>No data available for violin plot</p>
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
      {/* Outlier indicator */}
      {data.outlierCount && data.outlierCount > 0 && (
        <div className="text-center mt-2">
          <p className="text-text-tertiary text-xs">
            {data.outlierCount} statistical outliers above{" "}
            {((data.cappedAt ?? 0) / 1000).toFixed(1)}s not shown
          </p>
        </div>
      )}
    </div>
  );
};

export default ViolinPlot;
