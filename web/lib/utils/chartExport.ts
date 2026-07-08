// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

const triggerDownload = (href: string, filename: string) => {
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(href);
};

const loadImage = (src: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });

export function downloadCSV(
  rows: Record<string, unknown>[],
  filename: string
) {
  const [first] = rows;
  if (!first) return;
  const headers = Object.keys(first);
  const escape = (value: unknown) => {
    const s = String(value ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [
    headers.join(","),
    ...rows.map((row) => headers.map((h) => escape(row[h])).join(",")),
  ].join("\n");
  triggerDownload(
    URL.createObjectURL(new Blob([csv], { type: "text/csv" })),
    filename
  );
}

export interface LegendItem {
  label: string;
  color: string;
  /** Grayed out in the export, e.g. a series clipped off-chart by the zoom. */
  dimmed?: boolean;
}

export interface ChartPNGHeader {
  label?: string;
  title?: string;
  legend?: LegendItem[];
  /** Centered x-axis title drawn under the chart, for SVGs without one. */
  xLabel?: string;
  /** Headline stat drawn top-right, e.g. "Avg WER (all models)" / "3.6%". */
  stat?: { label: string; value: string };
  /** Mutates the cloned SVG before rasterizing, e.g. to label scatter dots. */
  annotate?: (clone: SVGSVGElement) => void;
}

const FONT = "ui-sans-serif, system-ui, sans-serif";
/** Outer margin between the canvas edge and every piece of content. */
const MARGIN = 24;
const LEGEND_ROW_HEIGHT = 20;

// Lay the legend out in wrapped rows of "swatch + label" items sized with
// canvas text metrics, so the header height is known before the canvas is
// created (resizing a canvas would wipe its state).
function legendRows(
  ctx: CanvasRenderingContext2D,
  legend: LegendItem[],
  maxWidth: number
) {
  ctx.font = `12px ${FONT}`;
  const rows: (LegendItem & { x: number })[][] = [];
  let row: (LegendItem & { x: number })[] = [];
  let x = 0;
  for (const entry of legend) {
    const itemWidth = 18 + ctx.measureText(entry.label).width + 18;
    if (row.length > 0 && x + itemWidth > maxWidth) {
      rows.push(row);
      row = [];
      x = 0;
    }
    row.push({ ...entry, x });
    x += itemWidth;
  }
  if (row.length > 0) rows.push(row);
  return rows;
}

const SVG_NS = "http://www.w3.org/2000/svg";

// Label text takes its dot's color so stacked labels stay attributable, but
// light series colors get darkened to keep the text readable on white.
const labelColor = (hex?: string) => {
  if (!hex || !/^#[0-9a-f]{6}$/i.test(hex)) return "#3d3d3d";
  const n = parseInt(hex.slice(1), 16);
  let [r, g, b] = [n >> 16, (n >> 8) & 255, n & 255];
  if ((0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.6) {
    [r, g, b] = [r, g, b].map((c) => Math.round(c * 0.65)) as [
      number,
      number,
      number,
    ];
  }
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
};

/**
 * Names each scatter dot in the export clone. Dots pair with data points by
 * identical (x, then y) ordering — cy is inverted in pixels. Labels avoid the
 * dots and each other; one squeezed out of a cluster drops below it with a
 * leader line back to its dot.
 */
export function labelScatterDots(
  clone: SVGSVGElement,
  points: { x: number; y: number; label: string; color?: string }[]
) {
  const pos = (el: Element, attr: string) => Number(el.getAttribute(attr));
  const circles = Array.from(clone.querySelectorAll('circle[r="6"]')).sort(
    (a, b) => pos(a, "cx") - pos(b, "cx") || pos(a, "cy") - pos(b, "cy")
  );
  const sorted = [...points].sort((a, b) => a.x - b.x || b.y - a.y);
  const width = Number(clone.getAttribute("width"));
  const height = Number(clone.getAttribute("height"));
  type Box = { x1: number; y1: number; x2: number; y2: number };
  const boxes: Box[] = circles.map((c) => ({
    x1: pos(c, "cx") - 8,
    y1: pos(c, "cy") - 8,
    x2: pos(c, "cx") + 8,
    y2: pos(c, "cy") + 8,
  }));
  const blocked = (box: Box) =>
    box.x1 < 2 ||
    box.x2 > width - 2 ||
    box.y1 < 2 ||
    box.y2 > height - 14 ||
    boxes.some(
      (b) => box.x1 < b.x2 && box.x2 > b.x1 && box.y1 < b.y2 && box.y2 > b.y1
    );
  circles.forEach((circle, index) => {
    const point = sorted[index];
    if (!point) return;
    const cx = pos(circle, "cx");
    const cy = pos(circle, "cy");
    const w = point.label.length * 6;
    const box = (x: number, y: number, anchor: string): Box => {
      const x1 = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2;
      return { x1, y1: y - 11, x2: x1 + w, y2: y + 3 };
    };
    const candidates: [number, number, string][] = [
      [cx + 10, cy + 4, "start"],
      [cx - 10, cy + 4, "end"],
      [cx, cy - 11, "middle"],
      [cx, cy + 18, "middle"],
      [cx + 10, cy - 7, "start"],
      [cx + 10, cy + 15, "start"],
      [cx - 10, cy - 7, "end"],
      [cx - 10, cy + 15, "end"],
    ];
    let placement = candidates.find(([x, y, a]) => !blocked(box(x, y, a)));
    if (!placement) {
      for (let step = 2; step < 10 && !placement; step++) {
        const y = cy + 5 + step * 13;
        if (!blocked(box(cx, y, "middle"))) placement = [cx, y, "middle"];
      }
      if (placement) {
        const leader = document.createElementNS(SVG_NS, "line");
        leader.setAttribute("x1", `${cx}`);
        leader.setAttribute("y1", `${cy + 7}`);
        leader.setAttribute("x2", `${cx}`);
        leader.setAttribute("y2", `${placement[1] - 9}`);
        leader.setAttribute("stroke", "#c8c6c2");
        leader.setAttribute("stroke-width", "1");
        clone.appendChild(leader);
      }
    }
    const [x, y, anchor] = placement ?? candidates[0]!;
    boxes.push(box(x, y, anchor));
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", `${x}`);
    text.setAttribute("y", `${y}`);
    if (anchor !== "start") text.setAttribute("text-anchor", anchor);
    text.setAttribute("font-size", "11");
    text.setAttribute("font-family", FONT);
    text.setAttribute("font-weight", "500");
    text.setAttribute("fill", labelColor(point.color));
    text.textContent = point.label;
    clone.appendChild(text);
  });
}

export async function downloadChartPNG(
  svg: SVGSVGElement,
  filename: string,
  header: ChartPNGHeader = {}
) {
  const svgRect = svg.getBoundingClientRect();
  const { width, height } = svgRect;
  // The rendered SVG often reserves empty space below its content (e.g. room
  // recharts keeps for its HTML legend); crop it so the image ends where the
  // chart does. The bound comes from text elements (axis ticks/titles are the
  // lowest visible content) — getBBox would count series paths that a zoom
  // clips out of view and extend far past the plot.
  let textBottom = 0;
  svg.querySelectorAll("text").forEach((text) => {
    textBottom = Math.max(textBottom, text.getBoundingClientRect().bottom);
  });
  const contentHeight =
    textBottom > svgRect.top
      ? Math.min(height, Math.max(1, textBottom - svgRect.top + 6))
      : height;
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("width", `${width}`);
  clone.setAttribute("height", `${height}`);
  header.annotate?.(clone);
  const chart = await loadImage(
    `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`
  );
  const logo = await loadImage("/coval-logo.svg");
  const logoWidth = 72;
  const logoHeight = (logo.height / logo.width) * logoWidth;
  const canvas = document.createElement("canvas");
  const measure = canvas.getContext("2d");
  if (!measure) return;
  const rows = legendRows(measure, header.legend ?? [], width);
  const titleBlock = Math.max(
    (header.label ? 20 : 0) + (header.title ? 30 : 0),
    header.stat ? 50 : 0
  );
  const headerBlock =
    titleBlock +
    rows.length * LEGEND_ROW_HEIGHT +
    (titleBlock > 0 || rows.length > 0 ? 12 : 0);
  // The bottom strip carries the optional x-axis title and the watermark.
  const bottomBlock = Math.max(header.xLabel ? 24 : 0, logoHeight + 10);
  const totalWidth = width + 2 * MARGIN;
  const totalHeight = MARGIN + headerBlock + contentHeight + bottomBlock + MARGIN / 2;
  canvas.width = totalWidth * 2;
  canvas.height = totalHeight * 2;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(2, 2);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, totalWidth, totalHeight);
  ctx.strokeStyle = "#dddbd4";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, totalWidth - 1, totalHeight - 1);
  let y = MARGIN;
  if (header.stat) {
    ctx.textAlign = "right";
    ctx.font = `13px ${FONT}`;
    ctx.fillStyle = "#515151";
    ctx.fillText(header.stat.label, totalWidth - MARGIN, y + 13);
    ctx.font = `700 24px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.fillStyle = "#0a0a0a";
    ctx.fillText(header.stat.value, totalWidth - MARGIN, y + 44);
    ctx.textAlign = "left";
  }
  if (header.label) {
    ctx.font = `13px ${FONT}`;
    ctx.fillStyle = "#515151";
    ctx.fillText(header.label, MARGIN, y + 13);
    y += 20;
  }
  if (header.title) {
    ctx.font = `600 20px ${FONT}`;
    ctx.fillStyle = "#0a0a0a";
    ctx.fillText(header.title, MARGIN, y + 20);
    y += 30;
  }
  y = MARGIN + titleBlock;
  ctx.font = `12px ${FONT}`;
  for (const row of rows) {
    for (const item of row) {
      ctx.globalAlpha = item.dimmed ? 0.35 : 1;
      ctx.fillStyle = item.color;
      ctx.fillRect(MARGIN + item.x, y + 5, 12, 12);
      ctx.fillStyle = "#0a0a0a";
      ctx.fillText(item.label, MARGIN + item.x + 18, y + 15);
    }
    y += LEGEND_ROW_HEIGHT;
  }
  ctx.globalAlpha = 1;
  const chartY = MARGIN + headerBlock;
  ctx.drawImage(
    chart,
    0,
    0,
    width,
    contentHeight,
    MARGIN,
    chartY,
    width,
    contentHeight
  );
  if (header.xLabel) {
    ctx.font = `12px ${FONT}`;
    ctx.fillStyle = "#515151";
    ctx.textAlign = "center";
    ctx.fillText(header.xLabel, totalWidth / 2, chartY + contentHeight + 16);
    ctx.textAlign = "left";
  }
  ctx.globalAlpha = 0.45;
  ctx.drawImage(
    logo,
    totalWidth - MARGIN - logoWidth,
    totalHeight - logoHeight - 8,
    logoWidth,
    logoHeight
  );
  canvas.toBlob((blob) => {
    if (blob) triggerDownload(URL.createObjectURL(blob), filename);
  });
}
