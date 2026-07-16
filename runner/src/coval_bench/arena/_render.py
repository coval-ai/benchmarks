# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free HTML/SVG rendering for the arena monitoring tools.

No matplotlib, no JS, no external assets: a heatmap is a colored ``<table>`` and
a line chart is hand-built ``<svg>``, so the output is one self-contained file an
operator can open in any browser. Pure string builders — trivially unit-testable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# Sequential white -> deep-blue ramp (low count -> high count).
_RAMP_LOW = (255, 255, 255)
_RAMP_HIGH = (8, 48, 107)
# CVD-validated 8-hue categorical palette; ordering is the safety mechanism.
# Beyond 8 series, identity is color+dash (never a 9th generated hue).
_SERIES_COLORS = (
    "#2a78d6",
    "#008300",
    "#e87ba4",
    "#eda100",
    "#1baf7a",
    "#eb6834",
    "#4a3aa7",
    "#e34948",
)
_SERIES_DASHES = ("", "6 3", "2 2", "8 3 2 3")


def _nice_step(span: float, target_ticks: int = 6) -> float:
    """A 1/2/5-scaled tick step giving roughly ``target_ticks`` labels."""
    raw = span / max(target_ticks, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if raw <= mult * magnitude:
            return mult * magnitude
    return 10.0 * magnitude


def _lerp_color(t: float) -> str:
    """Interpolate the white->blue ramp at ``t`` in [0, 1]; return ``#rrggbb``."""
    t = max(0.0, min(1.0, t))
    r, g, b = (round(lo + (hi - lo) * t) for lo, hi in zip(_RAMP_LOW, _RAMP_HIGH, strict=True))
    return f"#{r:02x}{g:02x}{b:02x}"


def document(title: str, *body: str) -> str:
    """Wrap rendered fragments in a minimal standalone HTML document."""
    style = (
        "body{font:13px/1.4 system-ui,sans-serif;margin:24px;color:#111}"
        "h1{font-size:18px}h2{font-size:14px;margin-top:28px}"
        "table{border-collapse:collapse}td,th{padding:3px 6px;text-align:center}"
        "th{font-weight:600;font-size:11px}"
        ".rowhdr{text-align:right;white-space:nowrap}"
        ".muted{color:#666}"
    )
    return (
        f"<!doctype html><meta charset=utf-8><title>{_esc(title)}</title>"
        f"<style>{style}</style><h1>{_esc(title)}</h1>" + "".join(body)
    )


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def heatmap_table(
    labels: Sequence[str],
    matrix: Sequence[Sequence[int]],
    *,
    caption: str = "",
) -> str:
    """Render a symmetric integer matrix as a color-graded ``<table>``.

    Diagonal cells are blanked. Color scales by count against the off-diagonal
    max, so empty inter-tier blocks (the SCALE-too-small "island" signal) read as
    white gaps. Zero cells show ``0`` on white — distinguishable from the blanked
    diagonal, which is intentional: a true zero is a matchup that never happened.
    """
    n = len(labels)
    peak = max(
        (matrix[i][j] for i in range(n) for j in range(n) if i != j),
        default=0,
    )
    head = "<tr><th></th>" + "".join(f"<th>{_esc(lbl)}</th>" for lbl in labels) + "</tr>"
    rows = []
    for i in range(n):
        cells = [f'<th class="rowhdr">{_esc(labels[i])}</th>']
        for j in range(n):
            if i == j:
                cells.append('<td style="background:#222"></td>')
                continue
            count = matrix[i][j]
            shade = (count / peak) if peak > 0 else 0.0
            fg = "#fff" if shade > 0.55 else "#111"
            cells.append(f'<td style="background:{_lerp_color(shade)};color:{fg}">{count}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    cap = f"<p class=muted>{_esc(caption)}</p>" if caption else ""
    return f"<table>{head}{''.join(rows)}</table>{cap}"


def line_chart(
    series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    x_label: str,
    y_label: str,
    ref_y: float | None = None,
    width: int = 720,
    height: int = 420,
) -> str:
    """Hand-build an SVG multi-line chart. ``series`` maps name -> (x, y) points.

    ``ref_y`` draws a dashed horizontal reference (e.g. the +/-9 Elo target). An
    empty ``series`` (or one with no finite points) renders an explanatory note
    instead of a degenerate axis.
    """
    pts = [(x, y) for s in series.values() for (x, y) in s]
    if not pts:
        return f"<p class=muted>no data to plot for {_esc(y_label)}</p>"
    pad = 48.0
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts] + ([ref_y] if ref_y is not None else [])
    # Anchor both axes at 0 when the data allows: votes and CI widths are
    # non-negative, and distance-to-zero is part of what the chart shows.
    x_min, x_max = min(0.0, min(xs)), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(ys)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    def px(x: float) -> float:
        return pad + (x - x_min) / x_span * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - y_min) / y_span * (height - 2 * pad)

    # Legend grows rightward from the plot edge; widen the canvas so the
    # color+dash swatch and long names are not clipped at the viewBox
    # (~6px/char at font 10, plus 28px for the swatch).
    label_px = max((len(name) for name in series), default=0) * 6 + 28
    canvas_w = width - pad + 12 + label_px
    parts = [f'<svg viewBox="0 0 {canvas_w:g} {height}" width="{canvas_w:g}" height="{height}">']
    parts.append(
        "<style>"
        "g.s polyline{stroke-width:2}"
        "g.s:hover polyline{stroke-width:3.5}"
        "svg:has(g.s:hover) g.s:not(:hover){opacity:.25}"
        "</style>"
    )

    x0, y0 = px(x_min), py(y_min)
    x_step = _nice_step(x_span)
    y_step = _nice_step(y_span)
    ticks: list[str] = []
    v = math.ceil(x_min / x_step) * x_step
    while v <= x_max + 1e-9:
        tx = px(v)
        ticks.append(
            f'<line x1="{tx:.1f}" y1="{pad}" x2="{tx:.1f}" y2="{y0:.1f}" stroke="#e1e0d9"/>'
            f'<line x1="{tx:.1f}" y1="{y0:.1f}" x2="{tx:.1f}" y2="{y0 + 5:.1f}" stroke="#c3c2b7"/>'
            f'<text x="{tx:.1f}" y="{y0 + 16:.1f}" text-anchor="middle" font-size="10" '
            f'fill="#898781">{v:g}</text>'
        )
        v += x_step
    if x_span <= 60:
        for unit in range(math.ceil(x_min), math.floor(x_max) + 1):
            tx = px(unit)
            ticks.append(
                f'<line x1="{tx:.1f}" y1="{y0:.1f}" x2="{tx:.1f}" y2="{y0 + 3:.1f}" '
                f'stroke="#c3c2b7"/>'
            )
    v = math.ceil(y_min / y_step) * y_step
    while v <= y_max + 1e-9:
        ty = py(v)
        ticks.append(
            f'<line x1="{pad}" y1="{ty:.1f}" x2="{width - pad}" y2="{ty:.1f}" stroke="#e1e0d9"/>'
            f'<line x1="{pad - 5}" y1="{ty:.1f}" x2="{pad}" y2="{ty:.1f}" stroke="#c3c2b7"/>'
            f'<text x="{pad - 8}" y="{ty + 3:.1f}" text-anchor="end" font-size="10" '
            f'fill="#898781">{v:g}</text>'
        )
        v += y_step
    parts.extend(ticks)

    parts.append(
        f'<line x1="{pad}" y1="{y0:.1f}" x2="{width - pad}" y2="{y0:.1f}" stroke="#c3c2b7"/>'
        f'<line x1="{x0:.1f}" y1="{pad}" x2="{x0:.1f}" y2="{y0:.1f}" stroke="#c3c2b7"/>'
    )
    parts.append(
        f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle" '
        f'font-size="12">{_esc(x_label)}</text>'
        f'<text x="14" y="{height / 2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 14 {height / 2})">{_esc(y_label)}</text>'
    )
    if ref_y is not None:
        ry = py(ref_y)
        parts.append(
            f'<line x1="{pad}" y1="{ry:.1f}" x2="{width - pad}" y2="{ry:.1f}" '
            f'stroke="#d03b3b" stroke-dasharray="4 3"/>'
            f'<text x="{width - pad}" y="{ry - 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="#d03b3b">target {ref_y:g}</text>'
        )
    for idx, (name, points) in enumerate(series.items()):
        color = _SERIES_COLORS[idx % len(_SERIES_COLORS)]
        dash = _SERIES_DASHES[(idx // len(_SERIES_COLORS)) % len(_SERIES_DASHES)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        markers = "".join(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}">'
            f"<title>{_esc(name)} — {x_label} {x:g}, {y_label} {y:.4g}</title></circle>"
            for x, y in points
        )
        parts.append(
            f'<g class="s"><polyline fill="none" stroke="{color}"{dash_attr} '
            f'points="{path}"><title>{_esc(name)}</title></polyline>{markers}</g>'
        )
        ly = 12 + idx * 14
        parts.append(
            f'<line x1="{width - pad + 6}" y1="{ly - 3}" x2="{width - pad + 26}" '
            f'y2="{ly - 3}" stroke="{color}"{dash_attr} stroke-width="2"/>'
            f'<text x="{width - pad + 30}" y="{ly}" font-size="10" '
            f'fill="#111">{_esc(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
