// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState, useEffect, useLayoutEffect, useRef } from "react";
import * as d3 from "d3";
import { BoxPlotProps } from "@/types/chart.types";
import { BoxPlotDataPoint } from "@/types/benchmark.types";
import {
  DedicatedBadge,
  useDedicatedInfoTip,
} from "@/components/shared/DedicatedInferenceInfo";
import { useThemeColors } from "@/hooks/useThemeColors";

// Match the text sizes on the timeline chart above: 14px axis labels, 12px
// tick/legend/category text.
const modelFontSize = 12;
const providerFontSize = 12;
const axisLabelFontSize = "14px";
const yAxisTickFontSize = "12px";
const modelLineHeight = 14;
// Bottom holds up to three label lines, the provider, the dedicated-inference
// marker, and the axis caption — 80px stacked marker over caption.
const margin = { top: 20, right: 8, bottom: 88, left: 40 };
const minSlotWidth = 48;
/** Share of a slot the label block may occupy; the rest is breathing room. */
const labelBandRatio = 0.82;
/** Floor the label text shrinks to before a slot has to widen instead. */
const minModelFont = 8;
/** Geist Mono advances a fixed 0.6em, so label widths are predictable. */
const monoAdvance = 0.6;

const BoxPlot: React.FC<BoxPlotProps> = ({
  data,
  width = 800,
  height = 500,
  getModelColor,
  getProviderForModel,
  normalizeModelName,
  dedicatedModels,
  isMobile = false
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const axisRef = useRef<SVGSVGElement>(null);
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
  // Explainer for the dedicated-inference marker under a model's axis label.
  const {
    iconHandlers: dedicatedIcon,
    overlay: dedicatedOverlay,
    dismiss: dismissDedicated,
  } = useDedicatedInfoTip(containerRef);
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

  // Handle responsive sizing — always fit the card's content box. Resizes
  // are debounced (matching the recharts containers' debounce) so a window
  // drag doesn't force a full d3 redraw per frame, and unchanged sizes bail
  // out before triggering a render.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const measure = () => {
      if (!containerRef.current) return;
      const width = containerRef.current.offsetWidth;
      setDimensions((d) =>
        d.width === width && d.height === height ? d : { width, height }
      );
    };
    const handleResize = () => {
      clearTimeout(timer);
      timer = setTimeout(measure, 50);
    };

    measure();
    window.addEventListener("resize", handleResize);

    // Track container size changes that happen without a window resize
    const resizeObserver = new ResizeObserver(handleResize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", handleResize);
      resizeObserver.disconnect();
    };
  }, [height]);

  // Guarantee every model a minimum-width slot; when they don't all fit the
  // container the chart scrolls horizontally (same pattern as the WER bar
  // chart) rather than crushing the axis labels into each other. This holds at
  // any viewport width — full screen, half, quarter, drag-resized, or mobile —
  // so wide desktops render flush while narrow ones scroll.
  // A slot also has to hold the widest line of its label at the floor font —
  // below that the text stops shrinking and spills into its neighbours.
  const slotWidth = Math.ceil(
    Math.max(
      minSlotWidth,
      ...data.data.map(
        (d) =>
          (Math.max(
            getProviderForModel(d.model).length,
            ...normalizeModelName(d.model).split(/[-_\s]/).map((w) => w.length)
          ) *
            minModelFont *
            monoAdvance) /
          labelBandRatio
      )
    )
  );
  const svgWidth = Math.max(
    dimensions.width,
    data.data.length * slotWidth + margin.left + margin.right
  );
  const scrollable = svgWidth > dimensions.width;

  // Layout effect so the d3 content redraws in the same frame the <svg>
  // resizes — the PNG export clones the SVG as soon as its width settles, and
  // a post-paint redraw would let it capture stale content in a resized box.
  useLayoutEffect(() => {
    if (!svgRef.current || data.data.length === 0) return;

    setTip(null);
    dismissDedicated();

    const chartWidth = svgWidth - margin.left - margin.right;
    const chartHeight = dimensions.height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Use the sorted order from the data (already sorted by median)
    const xScale = d3
      .scaleBand()
      .domain(data.data.map((d) => d.model))
      .range([6, chartWidth])
      .paddingInner(0.2)
      .paddingOuter(0);

    // Ticks land on 0.5 s multiples, with the step widened until the range
    // fits a readable tick count (phones get fewer — 19 labels of TTFT data
    // is noise). The domain snaps to those boundaries so the outermost
    // gridlines — the visible floor and ceiling — enclose the whiskers.
    const baseStepMs = 500;
    const maxTicks = isMobile ? 7 : 12;
    const tickStepMs =
      baseStepMs *
      Math.ceil(
        Math.max(data.globalMax - Math.max(0, data.globalMin), baseStepMs) /
          (baseStepMs * maxTicks)
      );
    const yMin = Math.max(0, Math.floor(data.globalMin / tickStepMs) * tickStepMs);
    let yMax = Math.ceil(data.globalMax / tickStepMs) * tickStepMs;
    if (yMax === yMin) yMax = yMin + tickStepMs;
    const yScale = d3
      .scaleLinear()
      .domain([yMin, yMax])
      .range([chartHeight, 0]);

    // The plot svg scrolls while the y-axis labels live in a separate fixed
    // svg to its left, so the scale stays visible at any scroll position.
    const g = svg
      .append("g")
      .attr("transform", `translate(0,${margin.top})`);

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

    yAxisGroup.selectAll("text").remove();

    yAxisGroup.select(".domain").remove();

    const axisSvg = d3.select(axisRef.current);
    axisSvg.selectAll("*").remove();
    const axisGroup = axisSvg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)
      .call(yAxis);

    axisGroup.selectAll("line").remove();
    axisGroup.select(".domain").remove();
    axisGroup.selectAll("text")
      .attr("fill", themeColors.axisText)
      .attr("font-size", yAxisTickFontSize);

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

    // Lucide's Server icon, drawn inline: it must live inside the axis SVG
    // (an HTML overlay would detach when the plot scrolls) and always show —
    // it is the dedicated-endpoint marker. Hovering or tapping it opens the
    // explainer, so the wrapper carries a generous transparent hit rect.
    const drawServerIcon = (x: number, y: number, size: number) => {
      const wrap = g
        .append("g")
        .attr("transform", `translate(${x},${y})`)
        .attr("role", "button")
        .attr("tabindex", 0)
        .attr("aria-label", "About dedicated inference");
      const icon = wrap
        .append("g")
        .attr("transform", `scale(${size / 24})`)
        .attr("stroke", themeColors.label)
        .attr("fill", "none")
        .attr("stroke-width", 2.4)
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round");
      icon.append("rect").attr("x", 2).attr("y", 2).attr("width", 20).attr("height", 8).attr("rx", 2);
      icon.append("rect").attr("x", 2).attr("y", 14).attr("width", 20).attr("height", 8).attr("rx", 2);
      icon.append("line").attr("x1", 6).attr("y1", 6).attr("x2", 6.01).attr("y2", 6);
      icon.append("line").attr("x1", 6).attr("y1", 18).attr("x2", 6.01).attr("y2", 18);
      wrap
        .append("rect")
        .attr("x", -10)
        .attr("y", -10)
        .attr("width", size + 20)
        .attr("height", size + 20)
        .attr("fill", "transparent");
      return wrap;
    };

    // Create custom wrapped text with model first
    {
      const labelMaxWidth = xScale.step() * labelBandRatio;

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

        const providerY = yPosition + modelLines.length * lineHeight + 1;
        providerNode
          .attr("font-size", `${providerFontSize * scale}px`)
          .attr("y", providerY);

        // Dedicated endpoints carry the server icon on its own line beneath
        // the label, centered in the slot; hover or tap opens the explainer.
        // Fixed size on purpose: crowded views shrink the label text, and the
        // marker must stay legible — it is the only dedicated cue at a glance.
        if (dedicatedModels?.has(model)) {
          const iconSize = 15;
          drawServerIcon(xPosition - iconSize / 2, providerY + 4, iconSize)
            .style("cursor", "help")
            .on("mouseenter", (e: React.MouseEvent) => dedicatedIcon.onMouseEnter(e))
            .on("mouseleave", dedicatedIcon.onMouseLeave)
            .on("click", (e: React.MouseEvent) => dedicatedIcon.onClick(e))
            .on("keydown", (e: React.KeyboardEvent) => dedicatedIcon.onKeyDown(e));
        }
      });
    }

    // X-axis label (the Y axis title is omitted — it's redundant with the
    // card heading).
    g.append("text")
      .attr(
        "transform",
        `translate(${chartWidth / 2}, ${chartHeight + margin.bottom - 6})`
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

      // Main box (IQR). Dedicated endpoints get a dashed border — the visual
      // cue that this box runs on reserved capacity, not the shared fleet.
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
        .attr("stroke-dasharray", dedicatedModels?.has(modelData.model) ? "5 3" : null)
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
      if (scrollable) {
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
  }, [data, dimensions.height, svgWidth, scrollable, getModelColor, getProviderForModel, normalizeModelName, dedicatedModels, dedicatedIcon, dismissDedicated, isMobile, themeColors]);

  // The measured container must always render — an early return here would
  // leave the sizing effect's ResizeObserver attached to nothing, freezing
  // dimensions at their initial value once data arrives.
  return (
    <div ref={containerRef} className="relative w-full" data-export-frame>
      {data.data.length === 0 ? (
        // Same height as the populated chart so toggling to a model with no
        // latency runs doesn't shift the sections below.
        <div
          className="flex items-center justify-center text-text-secondary"
          style={{ height: dimensions.height }}
        >
          <p>No data available for box plot</p>
        </div>
      ) : (
        <div className="flex">
          <svg
            ref={axisRef}
            width={margin.left}
            height={dimensions.height}
            className="shrink-0"
            data-chart-axis
          />
          <div
            className="min-w-0 flex-1 overflow-x-auto touch-manipulation select-none"
            onScroll={(e) => setScrollX(e.currentTarget.scrollLeft)}
          >
            <svg
              ref={svgRef}
              width={svgWidth - margin.left}
              height={dimensions.height}
              className="overflow-visible"
              style={{ background: "transparent" }}
            />
          </div>
        </div>
      )}
      {tip && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute z-10 whitespace-nowrap text-xs"
          style={{
            // When the chart scrolls the panel hangs from the top of the plot
            // — inside the card, clear of the heading above — and tracks its
            // column as the chart scrolls, so it never covers the box being
            // inspected and follows the finger while swiping.
            left: Math.min(
              Math.max(scrollable ? tip.x - scrollX : tip.x, 90),
              dimensions.width - 90
            ),
            top: scrollable ? margin.top : tip.yTop,
            transform: scrollable
              ? "translate(-50%, 0)"
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
            {getProviderForModel(tip.point.model)}{" "}
            {normalizeModelName(tip.point.model)}
          </p>
          {dedicatedModels?.has(tip.point.model) && <DedicatedBadge />}
          {(tip.pinned
            ? [
                ["Max", `${tip.point.stats.max.toFixed(0)}ms`],
                ["P95", `${tip.point.stats.p95.toFixed(0)}ms`],
                ["P75", `${tip.point.quartiles.q3.toFixed(0)}ms`],
                ["P50", `${tip.point.quartiles.median.toFixed(0)}ms`],
                ["P25", `${tip.point.quartiles.q1.toFixed(0)}ms`],
                ["Count", `${tip.point.stats.count}`]
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
      {dedicatedOverlay}
    </div>
  );
};

export default BoxPlot;
