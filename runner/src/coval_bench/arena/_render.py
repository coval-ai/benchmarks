# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free HTML/SVG rendering for the arena monitoring tools.

No matplotlib, no JS, no external assets: a heatmap is a colored ``<table>`` and
a line chart is hand-built ``<svg>``, so the output is one self-contained file an
operator can open in any browser. Pure string builders — trivially unit-testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

# Sequential white -> deep-blue ramp (low count -> high count).
_RAMP_LOW = (255, 255, 255)
_RAMP_HIGH = (8, 48, 107)
# Distinct hues cycled across series in a line chart.
_SERIES_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
)


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
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    def px(x: float) -> float:
        return pad + (x - x_min) / x_span * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - y_min) / y_span * (height - 2 * pad)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    parts.append(
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" '
        f'y2="{height - pad}" stroke="#999"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#999"/>'
    )
    parts.append(
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" '
        f'font-size="12">{_esc(x_label)}</text>'
        f'<text x="14" y="{height / 2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 14 {height / 2})">{_esc(y_label)}</text>'
    )
    if ref_y is not None:
        ry = py(ref_y)
        parts.append(
            f'<line x1="{pad}" y1="{ry}" x2="{width - pad}" y2="{ry}" '
            f'stroke="#c00" stroke-dasharray="4 3"/>'
            f'<text x="{width - pad}" y="{ry - 4}" text-anchor="end" '
            f'font-size="10" fill="#c00">{ref_y:g}</text>'
        )
    for idx, (name, points) in enumerate(series.items()):
        color = _SERIES_COLORS[idx % len(_SERIES_COLORS)]
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" points="{path}"/>')
        parts.append(
            f'<text x="{width - pad + 6}" y="{12 + idx * 14}" font-size="10" '
            f'fill="{color}">{_esc(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
