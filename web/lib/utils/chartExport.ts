// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { LIGHT_CHART_COLORS, type ThemeColors } from "@/hooks/useThemeColors";

// The export mirrors the on-screen theme. Chrome colours come from the same
// chart-* palette the charts render with; the shared light palette is the
// default used when a caller doesn't pass one.
const isDarkBg = (hex: string) => {
  if (!/^#[0-9a-f]{6}$/i.test(hex)) return false;
  const n = parseInt(hex.slice(1), 16);
  return (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255 < 0.5;
};

const triggerDownload = (href: string, filename: string) => {
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  a.click();
  // Browsers may read the blob URL after click() returns; revoking in the same
  // tick can abort the download, so defer it.
  window.setTimeout(() => URL.revokeObjectURL(href), 0);
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

/** Height of one wrapped title line (600 20px). */
const TITLE_LINE_HEIGHT = 28;

// Greedily wrap `text` into lines that fit `maxWidth` under the caller's current
// ctx.font. A single word wider than maxWidth stays on its own line (it can't be
// broken), so callers keep some headroom. Falls back to the whole string.
function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number
): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(candidate).width > maxWidth) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines.length ? lines : [text];
}

const SVG_NS = "http://www.w3.org/2000/svg";

// Label text takes its dot's color so stacked labels stay attributable, then is
// nudged for legibility on the export surface: bright colors darken on the light
// canvas, dark colors lighten on the dark canvas.
const labelColor = (hex?: string, dark = false) => {
  if (!hex || !/^#[0-9a-f]{6}$/i.test(hex)) return dark ? "#c7c2bc" : "#3d3d3d";
  const n = parseInt(hex.slice(1), 16);
  let [r, g, b] = [n >> 16, (n >> 8) & 255, n & 255];
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  if (!dark && lum > 0.6) {
    [r, g, b] = [r, g, b].map((c) => Math.round(c * 0.65)) as [number, number, number];
  } else if (dark && lum < 0.5) {
    [r, g, b] = [r, g, b].map((c) => Math.round(c + (255 - c) * 0.5)) as [number, number, number];
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
  points: { x: number; y: number; label: string; color?: string }[],
  colors: ThemeColors = LIGHT_CHART_COLORS
) {
  const dark = isDarkBg(colors.tooltipBg);
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
  // Stacked dots hide each other completely on screen; a rim in the canvas
  // color splits them into visible crescents so every label has a referent.
  circles.forEach((circle) => {
    if (!circle.getAttribute("stroke")) {
      circle.setAttribute("stroke", colors.tooltipBg);
      circle.setAttribute("stroke-width", "1.5");
    }
  });
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
    // In dense clusters, spiral outward in small rings so a squeezed-out label
    // still sits near its dot (its color keeps it attributable) rather than at
    // the end of a hard-to-follow leader line.
    for (let r = 22; r <= 58; r += 12) {
      for (let deg = 0; deg < 360; deg += 30) {
        const dx = Math.cos((deg * Math.PI) / 180);
        const dy = Math.sin((deg * Math.PI) / 180);
        candidates.push([
          cx + r * dx,
          cy + r * dy + 4,
          Math.abs(dx) < 0.4 ? "middle" : dx > 0 ? "start" : "end",
        ]);
      }
    }
    const placement = candidates.find(([x, y, a]) => !blocked(box(x, y, a)));
    // Last-resort fallback: no collision-free slot exists, so accept overlap
    // but keep the label horizontally in-bounds (never clipped at the edge).
    // Prefer the side with room; else clamp a centered label into the canvas.
    let fallback: [number, number, string];
    if (cx + 10 + w <= width - 2) fallback = [cx + 10, cy + 4, "start"];
    else if (cx - 10 - w >= 2) fallback = [cx - 10, cy + 4, "end"];
    else if (w >= width - 4)
      // Label is wider than the plot itself; anchor at the left edge so at least
      // its start is readable rather than clipping both ends.
      fallback = [2, cy + 4, "start"];
    else fallback = [Math.min(Math.max(cx, 2 + w / 2), width - 2 - w / 2), cy + 4, "middle"];
    const [x, y, anchor] = placement ?? fallback;
    const placed = box(x, y, anchor);
    boxes.push(placed);
    // A label pushed off its dot gets a short connector in the dot's own
    // color — grey lines are untrackable and were rejected in review.
    const tx = Math.min(Math.max(cx, placed.x1), placed.x2);
    const ty = Math.min(Math.max(cy, placed.y1), placed.y2);
    const dist = Math.hypot(tx - cx, ty - cy);
    if (dist > 16) {
      const [ux, uy] = [(tx - cx) / dist, (ty - cy) / dist];
      const tick = document.createElementNS(SVG_NS, "line");
      tick.setAttribute("x1", `${cx + ux * 8}`);
      tick.setAttribute("y1", `${cy + uy * 8}`);
      tick.setAttribute("x2", `${tx - ux * 2}`);
      tick.setAttribute("y2", `${ty - uy * 2}`);
      tick.setAttribute("stroke", labelColor(point.color, dark));
      tick.setAttribute("stroke-width", "1.5");
      tick.setAttribute("stroke-linecap", "round");
      clone.appendChild(tick);
    }
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", `${x}`);
    text.setAttribute("y", `${y}`);
    if (anchor !== "start") text.setAttribute("text-anchor", anchor);
    text.setAttribute("font-size", "11");
    text.setAttribute("font-family", FONT);
    text.setAttribute("font-weight", "500");
    text.setAttribute("fill", labelColor(point.color, dark));
    text.textContent = point.label;
    clone.appendChild(text);
  });
  // Record how far down the labels reached so the export crop keeps them.
  const bottom = boxes.reduce((max, b) => Math.max(max, b.y2), 0);
  clone.setAttribute("data-annotation-bottom", `${bottom}`);
}

export async function downloadChartPNG(
  svg: SVGSVGElement,
  filename: string,
  header: ChartPNGHeader = {},
  colors: ThemeColors = LIGHT_CHART_COLORS
): Promise<boolean> {
  const dark = isDarkBg(colors.tooltipBg);
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
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("width", `${width}`);
  clone.setAttribute("height", `${height}`);
  header.annotate?.(clone);
  // Annotations may land below the chart's own text (labelScatterDots records
  // how far down it drew); they must survive the crop too.
  const annotationBottom =
    Number(clone.getAttribute("data-annotation-bottom") ?? 0) + 4;
  const contentHeight = Math.min(
    height,
    Math.max(
      1,
      textBottom > svgRect.top ? textBottom - svgRect.top + 6 : height,
      annotationBottom
    )
  );
  let chart: HTMLImageElement;
  let logo: HTMLImageElement;
  try {
    chart = await loadImage(
      `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`
    );
    logo = await loadImage("/coval-logo.svg");
  } catch {
    return false;
  }
  const logoWidth = 72;
  const logoHeight = (logo.height / logo.width) * logoWidth;
  const canvas = document.createElement("canvas");
  const measure = canvas.getContext("2d");
  if (!measure) return false;
  const rows = legendRows(measure, header.legend ?? [], width);
  // Measure the headline stat first so we know how much width it claims.
  let statW = 0;
  if (header.stat) {
    measure.font = `13px ${FONT}`;
    const statLabelW = measure.measureText(header.stat.label).width;
    measure.font = `700 24px ui-monospace, SFMono-Regular, Menlo, monospace`;
    statW = Math.max(statLabelW, measure.measureText(header.stat.value).width);
  }
  // On a narrow (mobile) export the left title and right-aligned stat collide,
  // so stack the stat under the title when a legible title strip (≥140px) can't
  // sit beside it. Either way the title WRAPS into the width actually available,
  // so a long multi-word title wraps instead of running off the canvas edge.
  const statFitsBeside = header.stat ? statW + 16 + 140 <= width : false;
  const availableTitleWidth =
    width - (header.stat && statFitsBeside ? statW + 16 : 0);
  measure.font = `600 20px ${FONT}`;
  const titleLines = header.title
    ? wrapText(measure, header.title, availableTitleWidth)
    : [];
  const titleTextBlock =
    (header.label ? 20 : 0) + titleLines.length * TITLE_LINE_HEIGHT;
  const titleBlock = header.stat
    ? statFitsBeside
      ? Math.max(titleTextBlock, 50)
      : titleTextBlock + 50
    : titleTextBlock;
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
  if (!ctx) return false;
  ctx.scale(2, 2);
  ctx.fillStyle = colors.tooltipBg;
  ctx.fillRect(0, 0, totalWidth, totalHeight);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, totalWidth - 1, totalHeight - 1);
  let y = MARGIN;
  if (header.stat && statFitsBeside) {
    ctx.textAlign = "right";
    ctx.font = `13px ${FONT}`;
    ctx.fillStyle = colors.textSecondary;
    ctx.fillText(header.stat.label, totalWidth - MARGIN, y + 13);
    ctx.font = `700 24px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.fillStyle = colors.textPrimary;
    ctx.fillText(header.stat.value, totalWidth - MARGIN, y + 44);
    ctx.textAlign = "left";
  }
  if (header.label) {
    ctx.font = `13px ${FONT}`;
    ctx.fillStyle = colors.textSecondary;
    ctx.fillText(header.label, MARGIN, y + 13);
    y += 20;
  }
  if (titleLines.length) {
    ctx.font = `600 20px ${FONT}`;
    ctx.fillStyle = colors.textPrimary;
    for (const line of titleLines) {
      ctx.fillText(line, MARGIN, y + 20);
      y += TITLE_LINE_HEIGHT;
    }
  }
  if (header.stat && !statFitsBeside) {
    ctx.font = `13px ${FONT}`;
    ctx.fillStyle = colors.textSecondary;
    ctx.fillText(header.stat.label, MARGIN, y + 13);
    ctx.font = `700 24px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.fillStyle = colors.textPrimary;
    ctx.fillText(header.stat.value, MARGIN, y + 40);
    y += 50;
  }
  y = MARGIN + titleBlock;
  ctx.font = `12px ${FONT}`;
  for (const row of rows) {
    for (const item of row) {
      ctx.globalAlpha = item.dimmed ? 0.35 : 1;
      ctx.fillStyle = item.color;
      ctx.fillRect(MARGIN + item.x, y + 5, 12, 12);
      ctx.fillStyle = colors.textPrimary;
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
    ctx.fillStyle = colors.textSecondary;
    ctx.textAlign = "center";
    ctx.fillText(header.xLabel, totalWidth / 2, chartY + contentHeight + 16);
    ctx.textAlign = "left";
  }
  ctx.globalAlpha = 0.45;
  // The logo art is ink; the brand rule allows ink or white only, so flip it to
  // white on the dark canvas to keep the watermark readable. Tinted with a
  // composite pass rather than ctx.filter, which Safari doesn't implement.
  let mark: HTMLImageElement | HTMLCanvasElement = logo;
  if (dark) {
    const tint = document.createElement("canvas");
    tint.width = logoWidth * 2;
    tint.height = Math.ceil(logoHeight * 2);
    const tctx = tint.getContext("2d");
    if (tctx) {
      tctx.drawImage(logo, 0, 0, tint.width, tint.height);
      tctx.globalCompositeOperation = "source-in";
      tctx.fillStyle = "#ffffff";
      tctx.fillRect(0, 0, tint.width, tint.height);
      mark = tint;
    }
  }
  ctx.drawImage(
    mark,
    totalWidth - MARGIN - logoWidth,
    totalHeight - logoHeight - 8,
    logoWidth,
    logoHeight
  );
  canvas.toBlob((blob) => {
    if (blob) triggerDownload(URL.createObjectURL(blob), filename);
  });
  return true;
}
