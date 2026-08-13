"""WP-3 — cohort construction. The causality rule is the unit.

Step 3 of `docs/specs/wallet_positioning.md`. Cohorts are ranked on realised P&L over a window
ending **strictly before** `as_of`, re-ranked walk-forward. A ranking window that touches the
evaluation period is the A-2 error in new clothing: the combiner's weights were fitted three days
*after* the window they were scored on, and that alone produced its result (§5.1).

Contract encoded here:
  (a) **the decisive pair** — a wallet profitable only *after* `as_of` must not reach the top
      cohort, **and** an in-sample ranking must select it. Without that control the leakage test
      proves nothing: a test that passes because the ranker is broken looks identical to one that
      passes because the ranker is causal;
  (b) **deposits are not P&L.** Flows come from the WP-3 Part B ledger and are subtracted, so a
      wallet that merely funded its account does not rank as a winner — the failure that would
      make the whole cohort a list of people who wired money in;
  (c) **an unknown ledger type contaminates rather than silently passes** — the window's P&L is
      reported as contaminated, never as a clean number;
  (d) **normalisation** — cohort positioning is divided by cohort account value, so doubling one
      member's size does not double the cohort signal (one whale is not the cohort);
  (e) **rank stability is measured and reported, not assumed** — "the cohort is not a cohort" is
      failure mode 1 and the entire basis of the 30-day early kill;
  (f) an empty or single-member cohort is refused, not scored.
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

from wallets import cohorts as ch  # noqa: E402

HOUR = 3_600_000
T0 = 1_700_000_000_000


def _panel(series: dict[str, list[tuple[float, float]]], coin: str = "BTC") -> pd.DataFrame:
    """{wallet: [(account_value, upnl), ...]} -> a WP-2-shaped account panel, hourly."""
    rows = []
    for w, obs in series.items():
        for i, (av, upnl) in enumerate(obs):
            rows.append({"ts_ms": T0 + i * HOUR, "wallet": w, "coin": coin,
                         "account_value": av, "unrealized_pnl": upnl,
                         "position_value": 1000.0, "size": 1.0, "status": "ok"})
    return pd.DataFrame(rows)


def _flat(av: float, n: int) -> list[tuple[float, float]]:
    return [(av, 0.0)] * n


# ── (a) the decisive pair: leakage, and the control that makes it mean something ──

def _leakage_panel() -> pd.DataFrame:
    """LATE is flat for 10h then doubles; EARLY gains steadily in the first 10h."""
    late = _flat(1000.0, 10) + [(1000.0 + 200.0 * i, 0.0) for i in range(1, 11)]
    early = [(1000.0 + 50.0 * i, 0.0) for i in range(10)] + _flat(1500.0, 10)
    quiet = _flat(1000.0, 20)
    return _panel({"0x" + "1" * 40: late, "0x" + "2" * 40: early, "0x" + "3" * 40: quiet})


LATE, EARLY, QUIET = "0x" + "1" * 40, "0x" + "2" * 40, "0x" + "3" * 40


def test_a_wallet_profitable_only_after_as_of_is_not_in_the_top_cohort():
    """THE test. The ranking window must end strictly before `as_of`."""
    out = ch.rank_cohorts(_leakage_panel(), as_of=T0 + 10 * HOUR,
                          lookback_ms=10 * HOUR, k=1)
    assert LATE not in out["top"], \
        "a wallet whose gains are entirely after as_of leaked into the top cohort"
    assert EARLY in out["top"], "the genuinely-early winner should rank"


def test_control_an_in_sample_ranking_DOES_select_the_late_wallet():
    """Without this the leakage test proves nothing — it would pass on a broken ranker too."""
    out = ch.rank_cohorts(_leakage_panel(), as_of=T0 + 20 * HOUR,
                          lookback_ms=20 * HOUR, k=1)
    assert LATE in out["top"], \
        "the control failed: a window covering the late gains must select the late wallet"


def test_the_ranking_window_never_touches_as_of():
    out = ch.rank_cohorts(_leakage_panel(), as_of=T0 + 10 * HOUR,
                          lookback_ms=10 * HOUR, k=1)
    assert out["window"][1] <= T0 + 10 * HOUR
    assert out["window"][0] < out["window"][1]


# ── (b)+(c) flows are not P&L ────────────────────────────────────────────────────

def _deposit_entry(t_ms, usdc):
    return {"time": t_ms, "hash": f"0x{t_ms:064x}",
            "delta": {"type": "deposit", "usdc": str(usdc)}}


def test_a_deposit_does_not_register_as_pnl():
    """Account value rises by 500 purely because money was wired in."""
    panel = _panel({LATE: _flat(1000.0, 5) + _flat(1500.0, 5)})
    ledger = {LATE: [_deposit_entry(T0 + 5 * HOUR, 500.0)]}
    pnl, rep = ch.realised_pnl(panel, ledger, LATE, T0, T0 + 9 * HOUR)
    assert pnl == pytest.approx(0.0, abs=1e-6), f"a deposit was counted as P&L ({pnl})"
    assert rep["net_flow"] == pytest.approx(500.0)


def test_the_same_move_without_a_deposit_does_register():
    """The mirror — otherwise the test above passes on a function that always returns 0."""
    panel = _panel({LATE: _flat(1000.0, 5) + _flat(1500.0, 5)})
    pnl, _ = ch.realised_pnl(panel, {LATE: []}, LATE, T0, T0 + 9 * HOUR)
    assert pnl == pytest.approx(500.0), "genuine P&L was not recognised"


def test_unrealised_pnl_changes_are_not_realised_pnl():
    """Account value moved only because open positions marked to market."""
    panel = _panel({LATE: [(1000.0, 0.0)] * 5 + [(1200.0, 200.0)] * 5})
    pnl, _ = ch.realised_pnl(panel, {LATE: []}, LATE, T0, T0 + 9 * HOUR)
    assert pnl == pytest.approx(0.0, abs=1e-6)


def test_an_unknown_ledger_type_marks_the_window_contaminated():
    panel = _panel({LATE: _flat(1000.0, 5) + _flat(1500.0, 5)})
    ledger = {LATE: [{"time": T0 + 2 * HOUR, "hash": "0xabc",
                      "delta": {"type": "brandNewVenueThing", "usdc": "500.0"}}]}
    _, rep = ch.realised_pnl(panel, ledger, LATE, T0, T0 + 9 * HOUR)
    assert rep["contaminated"] is True
    assert rep["unknown_count"] == 1


# ── (d) normalisation ────────────────────────────────────────────────────────────

def test_doubling_one_member_does_not_double_the_cohort_signal():
    small = pd.DataFrame([
        {"ts_ms": T0, "wallet": EARLY, "coin": "BTC", "size": 1.0,
         "position_value": 1_000.0, "account_value": 10_000.0, "unrealized_pnl": 0.0,
         "status": "ok"},
        {"ts_ms": T0, "wallet": QUIET, "coin": "BTC", "size": 1.0,
         "position_value": 1_000.0, "account_value": 10_000.0, "unrealized_pnl": 0.0,
         "status": "ok"},
    ])
    big = small.copy()
    big.loc[big.wallet == EARLY, ["position_value", "account_value"]] *= 2

    a = ch.cohort_net_positioning(small, [EARLY, QUIET], "BTC", as_of=T0)
    b = ch.cohort_net_positioning(big, [EARLY, QUIET], "BTC", as_of=T0)
    assert b == pytest.approx(a, rel=1e-9), \
        "one member doubling in size changed the cohort signal — not normalised"


def test_positioning_is_signed():
    panel = pd.DataFrame([
        {"ts_ms": T0, "wallet": EARLY, "coin": "BTC", "size": -1.0,
         "position_value": 1_000.0, "account_value": 10_000.0, "unrealized_pnl": 0.0,
         "status": "ok"},
        {"ts_ms": T0, "wallet": QUIET, "coin": "BTC", "size": -1.0,
         "position_value": 1_000.0, "account_value": 10_000.0, "unrealized_pnl": 0.0,
         "status": "ok"},
    ])
    assert ch.cohort_net_positioning(panel, [EARLY, QUIET], "BTC", as_of=T0) < 0


# ── (e) rank stability is reported ───────────────────────────────────────────────

def test_a_stable_cohort_and_a_shuffled_one_are_distinguishable():
    stable = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    shuffled = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7]]
    s = ch.rank_stability([[f"0x{i:040x}" for i in g] for g in stable])
    u = ch.rank_stability([[f"0x{i:040x}" for i in g] for g in shuffled])
    assert s["mean_overlap"] == pytest.approx(1.0)
    assert u["mean_overlap"] < 0.5
    assert s["mean_overlap"] > u["mean_overlap"]


def test_stability_is_reported_even_when_it_is_bad():
    """The early kill needs the number, not a refusal to produce it."""
    out = ch.rank_stability([[f"0x{i:040x}" for i in g]
                             for g in ([1, 2, 3], [4, 5, 6])])
    assert "mean_overlap" in out and np.isfinite(out["mean_overlap"])
    assert out["n_rebalances"] == 2


# ── (f) degenerate cohorts are refused ───────────────────────────────────────────

@pytest.mark.parametrize("cohort", [[], ["0x" + "9" * 40]], ids=["empty", "single"])
def test_degenerate_cohort_is_refused_not_scored(cohort):
    panel = _panel({EARLY: _flat(1000.0, 3)})
    with pytest.raises(ValueError):
        ch.cohort_net_positioning(panel, cohort, "BTC", as_of=T0)


def test_ranking_refuses_when_too_few_wallets_have_history():
    panel = _panel({EARLY: _flat(1000.0, 3)})
    with pytest.raises(ValueError):
        ch.rank_cohorts(panel, as_of=T0 + 2 * HOUR, lookback_ms=2 * HOUR, k=2)


# ── hygiene ──────────────────────────────────────────────────────────────────────

def test_inputs_are_not_mutated():
    panel = _leakage_panel()
    before = panel.copy()
    ch.rank_cohorts(panel, as_of=T0 + 10 * HOUR, lookback_ms=10 * HOUR, k=1)
    pd.testing.assert_frame_equal(panel, before)


def test_deterministic():
    kw = dict(as_of=T0 + 10 * HOUR, lookback_ms=10 * HOUR, k=1)
    a = ch.rank_cohorts(_leakage_panel(), **kw)
    b = ch.rank_cohorts(_leakage_panel(), **kw)
    assert a == b
