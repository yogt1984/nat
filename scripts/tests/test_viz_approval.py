"""Planted tests for viz.approval — the T15 approval-evidence reductions (NAT6/NAT7).

Test-first (red). These pure reductions turn the OOS-validation state into the
series the `nat viz paper` / `nat viz portfolio` commands render: cumulative P&L,
IC-decay (rolling-sharpe proxy), per-signal risk, the G8 PASS/FAIL scorecard, and
the cross-signal correlation matrix (target < 0.35). All take a plain dict, so
they're exercised with synthetic OOS state — no filesystem, no real data.
"""

import math

import pytest

from viz.approval import (
    paper_pnl_series,
    ic_decay_series,
    per_signal_risk,
    g8_scorecard,
    signal_correlation_matrix,
    resolve_name,
)

_GATES = {
    "g8_min_sharpe_ratio": 0.5, "g8_max_daily_loss_pct": 2.0,
    "g8_max_ic_decay_pct": 50.0, "g8_min_days": 14,
}


def _state():
    """Synthetic OOS state mirroring data/oos_validation/state.json."""
    def block(baseline, daily):
        cum, run = [], 0.0
        for d in daily:
            run += d["total_net_bps"]
            cum.append({"date": d["date"], "cum_bps": round(run, 2)})
        sharpe = 1.5
        return {
            "baseline_sharpe": {"BTC": baseline},  # real data is a per-symbol dict
            "symbols": {
                "BTC": {
                    "daily": daily,
                    "metrics": {
                        "current_sharpe": sharpe,
                        "cumulative_pnl_bps": cum[-1]["cum_bps"] if cum else 0.0,
                        "max_drawdown_bps": -30.0,
                        "n_days": len(daily),
                        "degradation": 10.0,  # IC decay %
                        "rolling_sharpe_7d": [{"date": d["date"], "sharpe": sharpe} for d in daily],
                        "cumulative_pnl_series": cum,
                    },
                }
            },
        }

    d1 = [{"date": f"2026-05-{i:02d}", "symbol": "BTC", "total_net_bps": v, "max_loss_bps": -5.0}
          for i, v in zip(range(10, 25), [10, -5, 8, 12, -3, 9, 11, -4, 7, 13, -2, 6, 10, -3, 8])]
    # second signal: perfectly correlated with d1 (same daily series)
    d2 = [dict(x) for x in d1]
    # third signal: anti-correlated
    d3 = [{**x, "total_net_bps": -x["total_net_bps"]} for x in d1]
    return {
        "last_updated": "2026-05-25",
        "algos": {
            "3f": block(1.8, d1),
            "jump_detector": block(1.2, d2),
            "funding_reversion": block(1.0, d3),
        },
    }


# ---------------------------------------------------------------------------
# name resolution (lifecycle "3f_liquidity" ↔ OOS "3f")
# ---------------------------------------------------------------------------


class TestResolveName:
    def test_alias_3f(self):
        assert resolve_name(_state(), "3f_liquidity") == "3f"

    def test_exact(self):
        assert resolve_name(_state(), "jump_detector") == "jump_detector"

    def test_unknown(self):
        assert resolve_name(_state(), "nope") is None


# ---------------------------------------------------------------------------
# paper P&L + IC decay
# ---------------------------------------------------------------------------


class TestPaperSeries:
    def test_pnl_series_is_cumulative(self):
        s = paper_pnl_series(_state(), "3f_liquidity")
        assert len(s) == 15
        assert s[0]["cum_bps"] == pytest.approx(10.0)
        assert s[-1]["cum_bps"] == pytest.approx(s[-2]["cum_bps"] + 8.0)

    def test_ic_decay_series(self):
        s = ic_decay_series(_state(), "jump_detector")
        assert len(s) == 15 and "sharpe" in s[0] and "date" in s[0]

    def test_empty_for_unknown(self):
        assert paper_pnl_series(_state(), "nope") == []
        assert ic_decay_series(_state(), "nope") == []


# ---------------------------------------------------------------------------
# per-signal risk
# ---------------------------------------------------------------------------


class TestPerSignalRisk:
    def test_metrics_present(self):
        r = per_signal_risk(_state(), "3f_liquidity")
        assert r["sharpe"] == pytest.approx(1.5)
        assert r["max_dd_bps"] == pytest.approx(-30.0)
        assert r["n_days"] == 15

    def test_profit_factor(self):
        # wins=10+8+12+9+11+7+13+6+10+8=94 ; losses=5+3+4+2+3=17 ; PF=94/17≈5.53
        r = per_signal_risk(_state(), "3f_liquidity")
        assert r["profit_factor"] == pytest.approx(94 / 17, abs=0.05)

    def test_unknown_is_empty(self):
        assert per_signal_risk(_state(), "nope") == {}


# ---------------------------------------------------------------------------
# G8 scorecard (provisional, from OOS — reuses build_paper_report)
# ---------------------------------------------------------------------------


class TestG8Scorecard:
    def test_returns_five_criteria(self):
        sc = g8_scorecard(_state(), "3f_liquidity", _GATES)
        for k in ("gate_sharpe_within_2x", "gate_no_big_daily_loss",
                  "gate_ic_stable", "gate_infra_stable", "n_days"):
            assert k in sc

    def test_sharpe_ratio_gate(self):
        # paper_sharpe=1.5, baseline=1.8 -> ratio 0.83 >= 0.5 -> pass
        assert g8_scorecard(_state(), "3f_liquidity", _GATES)["gate_sharpe_within_2x"] is True

    def test_unknown_is_empty(self):
        assert g8_scorecard(_state(), "nope", _GATES) == {}


# ---------------------------------------------------------------------------
# cross-signal correlation matrix
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_shape_and_diagonal(self):
        m = signal_correlation_matrix(_state(), ["3f_liquidity", "jump_detector", "funding_reversion"])
        assert m["signals"] == ["3f_liquidity", "jump_detector", "funding_reversion"]
        assert len(m["matrix"]) == 3 and len(m["matrix"][0]) == 3
        for i in range(3):
            assert m["matrix"][i][i] == pytest.approx(1.0)

    def test_correlated_and_anticorrelated(self):
        m = signal_correlation_matrix(_state(), ["3f_liquidity", "jump_detector", "funding_reversion"])
        # 3f vs jump_detector are identical series → +1
        assert m["matrix"][0][1] == pytest.approx(1.0, abs=1e-6)
        # 3f vs funding_reversion are negated → -1
        assert m["matrix"][0][2] == pytest.approx(-1.0, abs=1e-6)

    def test_skips_unknown_signals(self):
        m = signal_correlation_matrix(_state(), ["3f_liquidity", "nope"])
        assert m["signals"] == ["3f_liquidity"]
