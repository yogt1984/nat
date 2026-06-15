"""Foundation tests for the `nat viz` terminal primitives (plan T7 / NAT3)."""

from __future__ import annotations

from viz.terminal import (
    sparkline, ic_color, bar, colorize, live_refresh,
    GREEN, RED, GREY, RESET,
)


def test_sparkline_ramp():
    s = sparkline([0, 1, 2, 3, 4, 5, 6, 7])
    assert len(s) == 8 and s[0] == "▁" and s[-1] == "█"


def test_sparkline_empty_and_all_nan():
    assert sparkline([]) == ""
    assert sparkline([float("nan"), float("nan")]) == ""


def test_sparkline_nan_renders_gap():
    s = sparkline([0.0, float("nan"), 1.0])
    assert len(s) == 3 and s[1] == " "


def test_sparkline_constant_is_uniform():
    assert len(set(sparkline([5, 5, 5]))) == 1


def test_ic_color_sign_and_magnitude():
    assert ic_color(None) == GREY
    assert ic_color(0.0) == GREY            # below the 0.02 floor
    assert GREEN in ic_color(0.06)          # strong long edge
    assert RED in ic_color(-0.06)           # strong short edge


def test_bar_clamps_to_width():
    assert bar(2.0, lo=-1, hi=1, width=10) == "█" * 10
    assert bar(-2.0, lo=-1, hi=1, width=10) == "·" * 10


def test_colorize_resets():
    assert colorize("x", GREEN).endswith(RESET)


def test_live_refresh_bounded(capsys):
    live_refresh(lambda: "tick", interval=0, iterations=3, clear=False)
    assert capsys.readouterr().out.count("tick") == 3
