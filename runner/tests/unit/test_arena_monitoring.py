# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the arena monitoring views (co-occurrence + convergence)."""

from __future__ import annotations

from datetime import datetime, timedelta

from coval_bench.arena._render import line_chart
from coval_bench.arena.monitoring import (
    _empty_inter_pairs,
    build_convergence,
    build_cooccurrence,
    render_convergence,
    render_cooccurrence,
)

_DT1 = datetime(2026, 1, 1)
_DT2 = _DT1 + timedelta(days=1)


# --- co-occurrence ---------------------------------------------------------


def test_cooccurrence_folds_ab_direction() -> None:
    rows = [("p", "a", "p", "b", 3), ("p", "b", "p", "a", 2)]
    ratings = {("p", "a"): 1600.0, ("p", "b"): 1500.0}
    cooc = build_cooccurrence(rows, ratings)
    # (a,b) and (b,a) collapse to one symmetric cell holding the full count.
    assert cooc.matrix[0][1] == 5
    assert cooc.matrix[1][0] == 5
    assert cooc.total == 5


def test_cooccurrence_axis_ordered_by_elo_desc() -> None:
    rows: list[tuple[str, str, str, str, int]] = []
    ratings = {("p", "low"): 1400.0, ("p", "high"): 1600.0, ("p", "mid"): 1500.0}
    cooc = build_cooccurrence(rows, ratings)
    assert cooc.labels == ["p/high", "p/mid", "p/low"]


def test_cooccurrence_elo_ties_break_deterministically() -> None:
    # Equal Elo, inserted b-before-a; tie-break is the (provider, model) key asc.
    ratings = {("p", "b"): 1500.0, ("p", "a"): 1500.0}
    cooc = build_cooccurrence([], ratings)
    assert cooc.labels == ["p/a", "p/b"]


def test_cooccurrence_rated_model_with_no_battles_is_an_empty_row() -> None:
    rows = [("p", "a", "p", "b", 4)]
    ratings = {("p", "a"): 1600.0, ("p", "b"): 1500.0, ("p", "c"): 1400.0}
    cooc = build_cooccurrence(rows, ratings)
    c = cooc.labels.index("p/c")
    assert all(cooc.matrix[c][j] == 0 for j in range(len(cooc.labels)))


def test_cooccurrence_unrated_model_appended_after_rated() -> None:
    rows = [("p", "a", "x", "z", 1)]
    ratings = {("p", "a"): 1600.0}
    cooc = build_cooccurrence(rows, ratings)
    assert cooc.labels == ["p/a", "x/z"]


def test_render_cooccurrence_renders_full_grid() -> None:
    rows = [("p", "a", "p", "b", 1)]
    ratings = {("p", "a"): 1600.0, ("p", "b"): 1500.0}
    html = render_cooccurrence(build_cooccurrence(rows, ratings))
    assert "<table>" in html
    assert html.count("<td") == 4  # n*n cells for a 2-model axis (incl. diagonal)
    assert ">1<" in html  # the folded battle count is rendered in its cell


def test_empty_inter_pairs_ranks_widest_gap_first() -> None:
    # 4 models, only the adjacent a-b pair has battled; every other pair is empty.
    rows = [("p", "a", "p", "b", 1)]
    ratings = {("p", "a"): 1600.0, ("p", "b"): 1500.0, ("p", "c"): 1400.0, ("p", "d"): 1300.0}
    cooc = build_cooccurrence(rows, ratings)
    empties = _empty_inter_pairs(cooc)
    # a (rank 0) vs d (rank 3) is the widest never-played matchup -> the island
    # signal -> must surface first.
    assert empties[0] == ("p/a", "p/d")
    assert ("p/a", "p/b") not in empties  # that pair *has* a battle


# --- convergence -----------------------------------------------------------


def test_convergence_time_to_threshold_is_first_crossing() -> None:
    rows = [
        (_DT1, "p", "a", 110, 50.0),
        (_DT2, "p", "a", 140, 8.0),
    ]
    conv = build_convergence(rows, threshold=9.0)
    assert conv.time_to_threshold["p/a"] == 140


def test_convergence_keeps_null_ci_model_off_the_line() -> None:
    rows: list[tuple[datetime, str, str, int, float | None]] = [
        (_DT1, "p", "b", 10, None),
    ]
    conv = build_convergence(rows)
    # null points never plot, but the model stays visible in the table.
    assert "p/b" not in conv.series
    assert conv.time_to_threshold["p/b"] is None
    assert conv.status["p/b"] == "preliminary"


def test_convergence_vote_floor_blocks_tiny_sample() -> None:
    # A tight CI on too few votes is a small-sample artifact, not convergence.
    rows = [(_DT1, "p", "a", 20, 5.0), (_DT2, "p", "a", 120, 5.0)]
    conv = build_convergence(rows, threshold=9.0, min_votes=100)
    assert conv.time_to_threshold["p/a"] == 120


def test_convergence_status_is_latest_ci_tier() -> None:
    rows = [(_DT1, "p", "a", 50, 80.0), (_DT2, "p", "a", 200, 20.0)]
    conv = build_convergence(rows)
    assert conv.status["p/a"] == "established"


def test_convergence_never_reached_is_none() -> None:
    rows = [(_DT1, "p", "a", 10, 50.0), (_DT2, "p", "a", 40, 20.0)]
    conv = build_convergence(rows, threshold=9.0)
    assert conv.time_to_threshold["p/a"] is None


def test_convergence_line_is_ordered_by_votes_not_time() -> None:
    # Pathological: votes decrease over time. The plotted line must still go
    # left-to-right on the votes axis.
    rows = [(_DT1, "p", "a", 140, 30.0), (_DT2, "p", "a", 110, 8.0)]
    conv = build_convergence(rows, threshold=9.0)
    xs = [x for x, _ in conv.series["p/a"]]
    assert xs == sorted(xs)
    # ...while time-to-threshold stays chronological (the DT2 snapshot).
    assert conv.time_to_threshold["p/a"] == 110


def test_render_convergence_renders_without_data() -> None:
    conv = build_convergence([])
    assert "<h1>" in render_convergence(conv)


def test_line_chart_widens_canvas_for_long_legend_labels() -> None:
    # Regression: legend text must fit inside the viewBox, not clip at the edge.
    short = line_chart({"a": [(0.0, 0.0)]}, x_label="x", y_label="y")
    long = line_chart(
        {"anthropic/claude-opus-4-some-very-long-name": [(0.0, 0.0)]},
        x_label="x",
        y_label="y",
    )
    assert _viewbox_width(long) > _viewbox_width(short)


def _viewbox_width(svg: str) -> float:
    inner = svg.split('viewBox="0 0 ', 1)[1]
    return float(inner.split(" ", 1)[0])
