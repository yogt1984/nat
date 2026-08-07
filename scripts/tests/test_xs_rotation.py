"""XS-10 — the rotation as a re-runnable unit.

§7.8's result was produced by a one-off script. A number nobody can re-run is a snapshot,
and §7.8's whole conclusion is that the question needs ~325 rebalances against the 83 it
had — so the strategy has to be callable on a growing archive.

Two properties carry it: the construction is actually beta-neutral (the entire point of
the rebuild — §7.7's tilt earned nothing and produced 80% of P&L variance), and the
extraction reproduces the recorded numbers rather than quietly becoming a different
strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from xs.rotation import run_rotation  # noqa: E402


def _panel(n_pairs=60, n_bars=900, seed=0, beta_spread=True):
    """Panel with a deliberate market factor and dispersed betas."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-05-01", periods=n_bars, freq="1h", tz="UTC")
    mkt = rng.normal(scale=0.004, size=n_bars)
    betas = np.linspace(0.3, 2.0, n_pairs) if beta_spread else np.ones(n_pairs)
    out = {}
    for j in range(n_pairs):
        r = betas[j] * mkt + rng.normal(scale=0.004, size=n_bars)
        out[f"P{j:02d}"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=ts)


def _costs(cols, bps=2.0):
    return pd.Series(bps, index=cols)


def test_the_portfolio_is_beta_neutral():
    """The rebuild's entire purpose: §7.7's -0.33 tilt earned nothing and dominated variance."""
    w = _panel()
    m = run_rotation(w, _costs(w.columns))
    assert m["n_periods"] > 5
    assert m["mean_abs_net_beta"] < 1e-6, (
        f"net beta {m['mean_abs_net_beta']} — the beta projection is not working"
    )


def test_weights_are_unit_gross_so_costs_are_comparable():
    w = _panel()
    m = run_rotation(w, _costs(w.columns))
    assert 0 < m["mean_turnover"] <= 2.0


def test_cost_stress_only_moves_cost():
    w = _panel()
    a = run_rotation(w, _costs(w.columns), cost_stress=1.0)
    b = run_rotation(w, _costs(w.columns), cost_stress=2.0)
    assert a["gross_total_pct"] == b["gross_total_pct"], "stress must not touch gross"
    # rel=1e-2, not 1e-6: cost_total_pct is rounded to 3 dp for reporting, so
    # 2*round(x) != round(2x) at these magnitudes. The invariant is the ratio, not the digits.
    assert b["cost_total_pct"] == pytest.approx(2 * a["cost_total_pct"], rel=1e-2)


def test_metrics_are_shaped_for_the_criteria_evaluator():
    from xs.trajectory import evaluate_criteria
    w = _panel()
    m = run_rotation(w, _costs(w.columns))
    m["dsr_p"], m["sign_stable_2x"] = 0.5, True
    passed, failed = evaluate_criteria(m)
    assert len(passed) + len(failed) == 6


def test_a_panel_too_thin_to_trade_reports_rather_than_crashes():
    w = _panel(n_pairs=10, n_bars=400)
    m = run_rotation(w, _costs(w.columns))
    assert m["n_periods"] == 0 and "reason" in m


def test_pairs_absent_early_do_not_break_the_run():
    """PROC-19 keeps the panel's holes; the rotation must tolerate them."""
    w = _panel()
    w.iloc[:400, 0] = np.nan          # a late listing
    m = run_rotation(w, _costs(w.columns))
    assert m["n_periods"] > 5


def test_a_hysteresis_band_changes_gross_not_only_cost():
    """Holding different positions must produce a different gross return.

    Regression for a real bug: gross was computed from the TARGET weights while the
    portfolio held the band-adjusted ones, so adding a band showed cost falling with gross
    untouched — free money, which is always a defect. Latent at band=0 where the two
    coincide.
    """
    w = _panel()
    base = run_rotation(w, _costs(w.columns), band=0.0)
    banded = run_rotation(w, _costs(w.columns), band=0.02, band_mode="edge")
    assert banded["mean_turnover"] < base["mean_turnover"], "the band did not reduce trading"
    assert banded["gross_total_pct"] != base["gross_total_pct"], (
        "gross is unchanged by a band that changed the held positions — gross is being "
        "priced on target weights rather than held ones"
    )
