"""TC-1 — planted tests for the trend-continuation study (15m/1h × universe).

The record refutes continuation at 1m/5m (34/36 cells negative) and finds
cross-sectional REVERSAL at daily rank (XS-3). The untested region is 15m-1h.
This study is PROC-20's momentum family swept across the candle universe; what the
tests plant is, as ever, the plumbing:

- a series built persistent must come back positive and significant; built
  anti-persistent, negative; a random walk, neither — the study finds what is
  there, in the direction it is there, and nothing in noise;
- the GATE cell is next-bar continuation (`p_continue_excess`), which is
  non-overlapping by construction — markouts at multi-bar horizons overlap and are
  recorded as descriptive only (VW-2's inflation lesson, §7.12);
- one sweep, corrected as one: BH-FDR runs across the whole (pair x interval x k)
  grid, so a marginal pair that would pass alone drowns among nulls;
- tz-aware candle timestamps must still produce calendar-day folds — a frame that
  cannot say which day a row is from gets refused upstream, and the study must not
  trip that refusal with the archive's own schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from exploration.trend_continuation_study import (  # noqa: E402
    GATE_METRIC, continuation_result, grid_fdr, sign_summary,
)
from processes.base import Finding, ProcessResult  # noqa: E402


def _candles(n: int, p_same: float, seed: int = 0, freq: str = "15min",
             sigma: float = 3e-3) -> pd.DataFrame:
    """Synthetic candles whose return SIGNS follow a Markov chain:
    P(sign_t = sign_{t-1}) = p_same. p_same > .5 is persistence, < .5 flip."""
    rng = np.random.default_rng(seed)
    sign = np.empty(n)
    sign[0] = 1.0
    for t in range(1, n):
        sign[t] = sign[t - 1] if rng.uniform() < p_same else -sign[t - 1]
    r = sign * np.abs(rng.normal(0, sigma, n))
    close = 100.0 * np.exp(np.cumsum(r))
    ts = pd.date_range("2026-06-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close,
                         "low": close, "close": close, "volume": 1.0})


def _run(p_same: float, seed: int = 0) -> ProcessResult:
    return continuation_result(_candles(2000, p_same, seed=seed), symbol="TEST",
                               interval="15m", n_shuffles=40, max_run_length=2,
                               day_shuffles=15, seed=0)


def _gate_cell(res: ProcessResult, k: int = 1) -> Finding:
    cells = [f for f in res.findings
             if f.metric == GATE_METRIC and f.extras.get("run_length") == k]
    assert cells, f"no {GATE_METRIC} cell for k={k}: {res.summary.get('error')}"
    return cells[0]


class TestPlantedDirection:
    def test_persistent_series_found_positive(self):
        f = _gate_cell(_run(p_same=0.75, seed=1))
        assert f.value > 0.15, f"planted persistence missed (excess={f.value})"
        assert f.extras["z"] >= 3.0

    def test_antipersistent_series_found_negative(self):
        f = _gate_cell(_run(p_same=0.25, seed=2))
        assert f.value < -0.15, f"planted flip missed (excess={f.value})"
        assert f.extras["z"] <= -3.0

    def test_random_walk_finds_nothing(self):
        f = _gate_cell(_run(p_same=0.5, seed=3))
        assert abs(f.extras["z"] or 0.0) < 3.0, \
            f"study finds continuation in a random walk (z={f.extras['z']})"
        assert not f.informative

    def test_tz_aware_timestamps_make_day_folds(self):
        res = _run(p_same=0.75, seed=4)
        assert res.summary.get("error") is None
        assert _gate_cell(res).extras["n_days"] >= 3   # 2000 x 15m ~ 20 days


class TestGateVsDescriptive:
    def test_gate_is_next_bar_continuation(self):
        """Markout cells exist (recorded) but only the non-overlapping next-bar
        cell feeds the sign summary — §7.12's overlap inflation must not gate."""
        res = _run(p_same=0.75, seed=5)
        assert any(f.metric == "markout_bps" for f in res.findings)
        rows = sign_summary([res])
        assert rows, "sign summary empty"
        assert all(r["metric"] == GATE_METRIC for r in rows)

    def test_sign_summary_counts(self):
        pos, neg = _run(p_same=0.75, seed=6), _run(p_same=0.25, seed=7)
        rows = sign_summary([pos, neg])
        k1 = [r for r in rows if r["run_length"] == 1]
        assert k1
        total = {(r["interval"],): r for r in k1}
        row = k1[0]
        assert row["n_pairs"] == 2
        assert row["n_pos"] == 1 and row["n_neg"] == 1


class TestGridFdr:
    def _finding(self, sym: str, p: float, value: float = 0.1) -> ProcessResult:
        r = ProcessResult(run_id=f"t_{sym}", process="persistence_stats",
                          kind="evaluation", symbol=sym, timeframe="15m", params={})
        r.findings.append(Finding(
            feature="momentum_run1_p_continue_excess", horizon="next_bar",
            metric="p_continue_excess", value=value, p_value=p, informative=True,
            extras={"run_length": 1, "z": 3.1, "n_events": 500,
                    "verdict": "durable", "n_days": 20,
                    "frac_days_informative": 0.7}))
        return r

    def test_one_sweep_corrected_as_one(self):
        strong = self._finding("STRONG", 1e-6, 0.2)
        marginal = self._finding("MARGINAL", 0.04)
        fillers = [self._finding(f"S{i}", 0.3 + 0.02 * i) for i in range(30)]
        report = grid_fdr([strong, marginal] + fillers)
        winners = {d["extras"].get("symbol") or d["feature"] for d in report.discoveries}
        syms = [d for d in report.discoveries]
        assert report.n_discoveries == 1
        assert strong.findings[0].informative is True
        assert marginal.findings[0].informative is False   # drowned by the grid
