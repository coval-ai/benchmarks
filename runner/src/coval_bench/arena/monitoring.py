# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Admin monitoring views over real arena vote data.

Two read-only diagnostics that tell us whether ``SCALE`` (the pairing volume
knob) is set sanely in production:

* **co-occurrence matrix** — who battled whom. Empty inter-tier blocks mean the
  battle graph has fragmented into islands (SCALE too small), which makes the
  Bradley-Terry fit's cross-tier Elo meaningless.
* **convergence** — CI half-width vs votes per model from snapshot history. Too
  many votes to reach the +/-9 Elo target means SCALE is too large (votes wasted
  on foregone-conclusion matchups).

Both reveal model identities, so these are operator/CLI tools, never the public
web surface. Pure functions take already-fetched rows; the CLI does the I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from coval_bench.arena import _render
from coval_bench.arena.rating import Status, classify_status

# Per-model CI half-width (Elo) we treat as "converged" for time-to-threshold.
# The established target band is +/-4..9 Elo; 9 is the loose end of that band.
DEFAULT_CONVERGED_CI = 9.0
# Cold-start vote floor: a tight CI on fewer votes is a small-sample artifact,
# not convergence (adaptive-pairing plan: ~100-300 votes/model).
DEFAULT_MIN_VOTES = 100

ModelKey = tuple[str, str]  # (provider, model)


def _label(key: ModelKey) -> str:
    return f"{key[0]}/{key[1]}"


# ---------------------------------------------------------------------------
# Co-occurrence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cooccurrence:
    """A symmetric who-battled-whom matrix over an ordered model axis."""

    labels: list[str]
    matrix: list[list[int]]
    total: int


def build_cooccurrence(
    rows: Sequence[tuple[str, str, str, str, int]],
    ratings: Mapping[ModelKey, float],
) -> Cooccurrence:
    """Fold per-direction battle counts into a symmetric matrix.

    ``rows`` are ``(provider_a, model_a, provider_b, model_b, count)`` straight
    from ``GROUP BY``. Because ``generate_battle`` randomizes which model lands in
    slot A vs B, the same matchup appears as both ``(X, Y)`` and ``(Y, X)``; we
    fold them together (``sorted`` pair key) so one matchup is one cell — not
    counting this would halve every cell and fake the island signal.

    Axis is ordered by ``rating_elo`` descending so tiers are visually adjacent
    and inter-tier gaps stand out. Rated models with zero battles still appear
    (a full empty row is itself a signal); models seen only in battles but absent
    from ``ratings`` are appended in sorted order.
    """
    pair_counts: dict[tuple[ModelKey, ModelKey], int] = {}
    seen: set[ModelKey] = set()
    for prov_a, model_a, prov_b, model_b, count in rows:
        a: ModelKey = (prov_a, model_a)
        b: ModelKey = (prov_b, model_b)
        seen.update((a, b))
        key = (a, b) if a <= b else (b, a)
        pair_counts[key] = pair_counts.get(key, 0) + count

    rated = sorted(ratings, key=lambda k: (-ratings[k], k))
    extra = sorted(seen - set(ratings))
    axis = rated + extra
    index = {key: i for i, key in enumerate(axis)}

    n = len(axis)
    matrix = [[0] * n for _ in range(n)]
    total = 0
    for (a, b), count in pair_counts.items():
        i, j = index[a], index[b]
        matrix[i][j] = count
        matrix[j][i] = count
        total += count
    return Cooccurrence(labels=[_label(k) for k in axis], matrix=matrix, total=total)


def _empty_inter_pairs(cooc: Cooccurrence, top_n: int = 8) -> list[tuple[str, str]]:
    """Widest-separated model pairs that have never battled — island candidates.

    "Widest-separated" is approximated by axis distance: the axis is rating-
    ordered, so a zero cell far from the diagonal is a top-vs-bottom matchup that
    never happened — exactly what fragments the graph.
    """
    n = len(cooc.labels)
    gaps = [
        (abs(i - j), cooc.labels[i], cooc.labels[j])
        for i in range(n)
        for j in range(i + 1, n)
        if cooc.matrix[i][j] == 0
    ]
    gaps.sort(reverse=True)
    return [(a, b) for _, a, b in gaps[:top_n]]


def render_cooccurrence(cooc: Cooccurrence) -> str:
    """Full standalone HTML document for the co-occurrence matrix."""
    empties = _empty_inter_pairs(cooc)
    note = (
        "Empty cells far from the diagonal are wide-gap matchups that never "
        "happened — if there are blocks of them, SCALE is too small (islands)."
    )
    empty_list = (
        "<h2>Widest never-played matchups</h2><ul>"
        + "".join(f"<li>{_render._esc(a)} &times; {_render._esc(b)}</li>" for a, b in empties)
        + "</ul>"
        if empties
        else "<p class=muted>Every adjacent-or-wider matchup has at least one battle.</p>"
    )
    table = _render.heatmap_table(
        cooc.labels,
        cooc.matrix,
        caption=f"{cooc.total} battles · axis ordered by Elo (strongest first). {note}",
    )
    return _render.document("Arena co-occurrence", table, empty_list)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Convergence:
    """Per-model CI-vs-votes trajectories plus time-to-threshold."""

    # model label -> ordered (votes_total, ci_half_width) points (null CI omitted)
    series: dict[str, list[tuple[float, float]]]
    # model label -> votes at first snapshot meeting BOTH bars (or None)
    time_to_threshold: dict[str, int | None]
    # model label -> current confidence tier from its latest CI half-width
    status: dict[str, Status]
    threshold: float
    min_votes: int


def build_convergence(
    rows: Sequence[tuple[datetime, str, str, int, float | None]],
    *,
    threshold: float = DEFAULT_CONVERGED_CI,
    min_votes: int = DEFAULT_MIN_VOTES,
) -> Convergence:
    """Group snapshot history ``(computed_at, provider, model, votes_total,
    ci_half_width)`` into per-model trajectories.

    Every model appears in ``time_to_threshold`` and ``status`` even if its CI is
    always null; only null points drop from the plotted ``series``. Converged =
    first snapshot with CI <= ``threshold`` Elo AND votes >= ``min_votes`` (the
    floor blocks tiny-sample false positives). ``status`` is ``classify_status``
    on the latest CI.
    """
    by_model: dict[str, list[tuple[datetime, int, float | None]]] = {}
    for computed_at, prov, model, votes, ci in rows:
        by_model.setdefault(_label((prov, model)), []).append((computed_at, votes, ci))

    series: dict[str, list[tuple[float, float]]] = {}
    ttt: dict[str, int | None] = {}
    status: dict[str, Status] = {}
    for label, points in by_model.items():
        points.sort(key=lambda p: p[0])
        # first chronological snapshot clearing both bars
        crossed = next(
            (
                votes
                for _, votes, ci in points
                if ci is not None and ci <= threshold and votes >= min_votes
            ),
            None,
        )
        ttt[label] = crossed
        # latest CI drives the tier (None -> preliminary)
        status[label] = classify_status(points[-1][2])
        # drop null points; order by votes for a monotonic x-axis
        pts = [(float(votes), ci) for _, votes, ci in points if ci is not None]
        if pts:
            series[label] = sorted(pts, key=lambda p: p[0])
    return Convergence(
        series=series,
        time_to_threshold=ttt,
        status=status,
        threshold=threshold,
        min_votes=min_votes,
    )


def render_convergence(conv: Convergence) -> str:
    """Full standalone HTML document for the convergence view."""
    chart = _render.line_chart(
        conv.series,
        x_label="votes_total",
        y_label="ci_half_width (Elo)",
        ref_y=conv.threshold,
    )

    def _row(label: str) -> str:
        votes = conv.time_to_threshold[label]
        cell = "not reached" if votes is None else str(votes)
        return (
            f"<tr><th class=rowhdr>{_render._esc(label)}</th>"
            f"<td>{conv.status[label]}</td><td>{cell}</td></tr>"
        )

    table = (
        f"<h2>Convergence (target +/-{conv.threshold:g} Elo, min {conv.min_votes} votes)</h2>"
        "<table><tr><th></th><th>tier</th><th>votes to converge</th></tr>"
        + "".join(_row(label) for label in sorted(conv.time_to_threshold))
        + "</table>"
        "<p class=muted>&gt;5,000 votes to converge ⇒ SCALE too large "
        "(votes wasted on blowouts).</p>"
    )
    return _render.document("Arena convergence", chart, table)
