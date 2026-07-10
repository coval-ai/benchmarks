// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import { BoxPlotProps } from "@/types/chart.types";
import { BoxPlotDataPoint } from "@/types/benchmark.types";
import { useThemeColors } from "@/hooks/useThemeColors";

// Match the text sizes on the timeline chart above: 14px axis labels, 12px
// tick/legend/category text.
const modelFontSize = 12;
const providerFontSize = 12;
const axisLabelFontSize = "14px";
const yAxisTickFontSize = "12px";
const modelLineHeight = 14;
const margin = { top: 20, right: 8, bottom: 80, left: 40 };
const mobileSlotWidth = 48;

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
  const [tip, setTip] = useState<{
    point: BoxPlotDataPoint;
    x: number;
    yTop: number;
    pinned: boolean;
  } | null>(null);
  const [scrollX, setScrollX] = useState(0);
  const themeColors = useThemeColors();

  useEffect(() => {
    if (!tip?.pinned) return;
    const onClick = () => setTip(null);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setTip(null);
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [tip]);

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

  // On mobile every model gets a guaranteed slot and the chart scrolls
  // horizontally (same pattern as the WER bar chart) — wide enough slots
  // re-enable the desktop axis labels and make each column a real touch
  // target.
  const svgWidth = isMobile
    ? Math.max(
        dimensions.width,
        data.data.length * mobileSlotWidth + margin.left + margin.right
      )
    : dimensions.width;

  useEffect(() => {
    if (!svgRef.current || data.data.length === 0) return;

    setTip(null);

    const chartWidth = svgWidth - margin.left - margin.right;
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

    // Create custom wrapped text with model first
    {
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

      // Full-height invisible hit area so a finger can land anywhere in the
      // model's column, not just on the thin strokes.
      if (isMobile) {
        boxGroup
          .append("rect")
          .attr("x", centerX - xScale.step() / 2)
          .attr("y", 0)
          .attr("width", xScale.step())
          .attr("height", chartHeight)
          .attr("fill", "transparent");
      }

      // Hover shows a compact name + median tooltip above the whiskers;
      // clicking pins that same tooltip and populates it with the full
      // stats so it never chases the cursor or blocks neighboring boxes.
      // On mobile there is no hover: a tap pins the full stats directly.
      const anchor = {
        point: modelData,
        x: margin.left + centerX,
        yTop: margin.top + yScale(max),
      };

      boxGroup.style("cursor", "pointer");

      if (!isMobile) {
        boxGroup
          .on("mouseenter", function () {
            d3.select(this)
              .select("rect")
              .attr("fill-opacity", 0.5);

            setTip((t) =>
              t?.pinned && t.point.model === modelData.model
                ? t
                : { ...anchor, pinned: false }
            );
          })
          .on("mouseleave", function () {
            d3.select(this)
              .select("rect")
              .attr("fill-opacity", 0.3);
          });
      }

      boxGroup
        .on("click", (event: MouseEvent) => {
          event.stopPropagation();
          setTip((t) =>
            t?.pinned && t.point.model === modelData.model
              ? null
              : { ...anchor, pinned: true }
          );
        });
    });

    svg.on("mouseleave", () => setTip((t) => (t?.pinned ? t : null)));
  }, [data, dimensions, svgWidth, getModelColor, getProviderForModel, normalizeModelName, isMobile, themeColors]);

  // The measured container must always render — an early return here would
  // leave the sizing effect's ResizeObserver attached to nothing, freezing
  // dimensions at their initial value once data arrives.
  return (
    <div ref={containerRef} className="relative w-full">
      {data.data.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-text-secondary">
          <p>No data available for box plot</p>
        </div>
      ) : (
        <div
          className="overflow-x-auto touch-manipulation select-none"
          onScroll={(e) => setScrollX(e.currentTarget.scrollLeft)}
        >
          <svg
            ref={svgRef}
            width={svgWidth}
            height={dimensions.height}
            className="overflow-visible"
            style={{ background: "transparent" }}
          />
        </div>
      )}
      {tip && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute z-10 whitespace-nowrap text-xs"
          style={{
            // On mobile the panel sits above the plot and tracks its column
            // as the chart scrolls, so it never covers the box being
            // inspected and follows the finger while swiping.
            left: Math.min(
              Math.max(isMobile ? tip.x - scrollX : tip.x, 90),
              dimensions.width - 90
            ),
            top: isMobile ? margin.top : tip.yTop,
            transform: isMobile
              ? "translate(-50%, -100%)"
              : "translate(-50%, calc(-100% - 8px))",
            transition: "left 150ms ease-out, top 150ms ease-out",
            pointerEvents: tip.pinned ? "auto" : "none",
            backgroundColor: "var(--color-surface-tooltip)",
            border: `1px solid ${getModelColor(tip.point.model)}`,
            borderRadius: "8px",
            padding: tip.pinned ? "6px 24px 6px 10px" : "6px 10px",
            boxShadow: "0 2px 8px rgba(10, 10, 10, 0.08)"
          }}
        >
          {tip.pinned && (
            <button
              aria-label="Close"
              onClick={() => setTip(null)}
              style={{
                position: "absolute",
                top: "4px",
                right: "6px",
                background: "none",
                border: "none",
                padding: 0,
                cursor: "pointer",
                lineHeight: 1,
                color: "var(--color-text-on-tooltip-secondary)"
              }}
            >
              ✕
            </button>
          )}
          <p
            style={{
              margin: 0,
              fontWeight: "bold",
              color: "var(--color-text-on-tooltip)"
            }}
          >
            {tip.point.model}
          </p>
          {(tip.pinned
            ? [
                ["Count", `${tip.point.stats.count}`],
                ["Mean", `${tip.point.stats.mean.toFixed(0)}ms`],
                ["Median", `${tip.point.quartiles.median.toFixed(0)}ms`],
                ["Std", `${tip.point.stats.std.toFixed(0)}ms`]
              ]
            : [
                [
                  "Median",
                  `${tip.point.quartiles.median.toFixed(0)}ms · click for details`
                ]
              ]
          ).map(([label, value]) => (
            <p
              key={label}
              style={{
                margin: 0,
                color: "var(--color-text-on-tooltip-secondary)"
              }}
            >
              {label}: {value}
            </p>
          ))}
        </div>
      )}
    </div>
  );
};

export default BoxPlot;
