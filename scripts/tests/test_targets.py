"""PROC-17 — the target as a first-class node. Adversarial: leakage is the enemy.

Today three processes (`ic_horizon`, `ml_importance`, `info_theory`) each re-derive the
same five decisions from a stringly-typed `target_col`: resolve it, check it exists,
materialise it, exclude the columns that leak it, and pick the right gate. Five copies of
a leakage rule is five chances to forget it — and forgetting it once produces a finding
that "predicts" a label from the label's own sibling column.

Contract encoded here:
  (a) one resolution rule (explicit param > context > none), so the three processes cannot
      drift apart;
  (b) a missing target column is an ERROR, never a silent fall back to forward returns —
      a finding stamped with the wrong target is worse than no finding;
  (c) the target owns its leakage set: itself, its siblings, and the price columns that are
      trivially correlated with a return;
  (d) gate selection follows the target's kind — a label is not a tradeable return, so the
      fee-based `i_min` gate does not apply to it (`info_theory._evaluate_label`);
  (e) forward returns are causal: r(t) uses prices t→t+h and the last h rows are NaN;
  (f) signedness is a property of the realised target, and it is the precondition for
      PROC-1's polarity — an unsigned target can never yield a trading direction.
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

from processes.base import ProcessContext  # noqa: E402


def _ctx(**over) -> ProcessContext:
    kw = dict(symbol="BTC", timeframe="15min", price_col="raw_midprice",
              horizons={"1m": 4, "5m": 20}, costs={})
    kw.update(over)
    return ProcessContext(**kw)


def _bars(n=200, with_labels=True) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "raw_midprice": 100 + np.cumsum(rng.normal(scale=0.05, size=n)),
        "raw_microprice": 100 + np.cumsum(rng.normal(scale=0.05, size=n)),
        "imbalance_qty_l1": rng.normal(size=n),
        "ent_book_shape": rng.uniform(size=n),
    })
    if with_labels:
        df["tb_label"] = rng.choice([-1.0, 0.0, 1.0], size=n)
        df["tb_ret"] = rng.normal(scale=0.001, size=n)
        df["tb_hit_bars"] = rng.integers(1, 20, size=n).astype(float)
    return df


# ── (a) one resolution rule ──────────────────────────────────────────────────────
class TestResolution:
    def test_explicit_param_beats_context(self):
        from processes.targets import resolve_target_col
        assert resolve_target_col({"target_col": "tb_label"}, _ctx(target_col="other")) == "tb_label"

    def test_context_used_when_param_absent(self):
        from processes.targets import resolve_target_col
        for params in ({}, {"target_col": None}, {"target_col": ""}):
            assert resolve_target_col(params, _ctx(target_col="tb_label")) == "tb_label"

    def test_none_means_forward_returns(self):
        from processes.targets import resolve_target_col
        assert resolve_target_col({}, _ctx()) is None


# ── (b) a missing column must not degrade into a different target ────────────────
class TestMissingColumnIsAnError:
    def test_absent_target_raises(self):
        from processes.targets import TargetNotFound, resolve_targets
        with pytest.raises(TargetNotFound, match="never_computed"):
            resolve_targets(_bars(), _ctx(), {"target_col": "never_computed"})

    def test_absent_target_does_not_silently_return_forward_returns(self):
        from processes.targets import TargetNotFound, resolve_targets
        try:
            out = resolve_targets(_bars(), _ctx(), {"target_col": "never_computed"})
        except TargetNotFound:
            return
        pytest.fail(f"silently produced {[t.name for t in out]} instead of raising")

    def test_all_nan_target_is_refused(self):
        from processes.targets import TargetNotFound, resolve_targets
        bars = _bars()
        bars["tb_label"] = np.nan
        with pytest.raises(TargetNotFound, match="no finite"):
            resolve_targets(bars, _ctx(), {"target_col": "tb_label"})

    def test_constant_target_is_refused(self):
        """A constant target has zero entropy — MI against it is 0 by construction."""
        from processes.targets import TargetNotFound, resolve_targets
        bars = _bars()
        bars["tb_label"] = 1.0
        with pytest.raises(TargetNotFound, match="constant"):
            resolve_targets(bars, _ctx(), {"target_col": "tb_label"})


# ── (c) the target owns its leakage set ──────────────────────────────────────────
class TestLeakage:
    def test_barrier_label_excludes_all_its_siblings(self):
        from processes.targets import resolve_targets
        bars = _bars()
        target = resolve_targets(bars, _ctx(), {"target_col": "tb_label"})[0]
        leaked = target.leakage_columns(bars)
        assert {"tb_label", "tb_ret", "tb_hit_bars"} <= leaked

    def test_feature_columns_never_contain_the_target_or_its_siblings(self):
        from processes.targets import feature_columns, resolve_targets
        bars = _bars()
        target = resolve_targets(bars, _ctx(), {"target_col": "tb_label"})[0]
        feats = feature_columns(bars, list(bars.columns), target)
        assert not any(c.startswith("tb_") for c in feats), feats
        assert "imbalance_qty_l1" in feats and "ent_book_shape" in feats

    def test_price_columns_are_leakage_for_forward_returns(self):
        """A forward return is a price ratio; scoring price against it is circular."""
        from processes.targets import feature_columns, resolve_targets
        bars = _bars()
        target = resolve_targets(bars, _ctx(), {})[0]
        feats = feature_columns(bars, list(bars.columns), target)
        assert "raw_midprice" not in feats and "raw_microprice" not in feats

    def test_a_non_barrier_label_excludes_only_itself(self):
        from processes.targets import feature_columns, resolve_targets
        bars = _bars()
        bars["custom_label"] = np.sign(bars["imbalance_qty_l1"].to_numpy())
        target = resolve_targets(bars, _ctx(), {"target_col": "custom_label"})[0]
        feats = feature_columns(bars, list(bars.columns), target)
        assert "custom_label" not in feats
        assert "tb_label" in feats, "unrelated columns must survive"


# ── (d) gate selection follows the kind ──────────────────────────────────────────
class TestGateSelection:
    def test_label_is_not_cost_gated(self):
        from processes.targets import resolve_targets
        t = resolve_targets(_bars(), _ctx(), {"target_col": "tb_label"})[0]
        assert t.kind == "label" and t.cost_gated is False
        assert t.gate == "null_z"

    def test_forward_return_is_cost_gated(self):
        from processes.targets import resolve_targets
        t = resolve_targets(_bars(), _ctx(), {})[0]
        assert t.kind == "forward_return" and t.cost_gated is True
        assert t.gate == "fee"

    def test_horizon_naming_matches_the_existing_convention(self):
        from processes.targets import resolve_targets
        label = resolve_targets(_bars(), _ctx(), {"target_col": "tb_label"})
        assert [t.horizon_name for t in label] == ["label"]
        assert label[0].horizon_bars == 0
        rets = resolve_targets(_bars(), _ctx(), {})
        assert [t.horizon_name for t in rets] == ["1m", "5m"]
        assert [t.horizon_bars for t in rets] == [4, 20]


# ── (e) causality ────────────────────────────────────────────────────────────────
class TestForwardReturnsAreCausal:
    def test_last_h_rows_are_nan_and_sign_is_recoverable(self):
        from processes.targets import resolve_targets
        n, h = 100, 4
        bars = pd.DataFrame({"raw_midprice": np.linspace(100, 110, n),
                             "imbalance_qty_l1": np.zeros(n)})
        t = resolve_targets(bars, _ctx(horizons={"h": h}), {})[0]
        v = t.values(bars)
        assert np.isnan(v[-h:]).all(), "forward return must not exist past the data"
        assert np.isfinite(v[:-h]).all()
        assert (v[:-h] > 0).all(), "monotone ramp must yield positive forward returns"

    def test_values_do_not_mutate_the_frame(self):
        from processes.targets import resolve_targets
        bars = _bars()
        before = bars.copy()
        for t in resolve_targets(bars, _ctx(), {}):
            t.values(bars)
        pd.testing.assert_frame_equal(bars, before)

    def test_label_values_are_the_column_verbatim(self):
        from processes.targets import resolve_targets
        bars = _bars()
        t = resolve_targets(bars, _ctx(), {"target_col": "tb_label"})[0]
        np.testing.assert_array_equal(t.values(bars), bars["tb_label"].to_numpy(float))


# ── (f) signedness — the precondition for PROC-1 polarity ────────────────────────
class TestSignedness:
    def test_three_class_barrier_label_is_signed(self):
        from processes.targets import resolve_targets
        t = resolve_targets(_bars(), _ctx(), {"target_col": "tb_label"})[0]
        assert t.signed is True

    def test_forward_returns_are_signed(self):
        from processes.targets import resolve_targets
        assert resolve_targets(_bars(), _ctx(), {})[0].signed is True

    def test_a_magnitude_target_is_unsigned(self):
        from processes.targets import resolve_targets
        bars = _bars()
        bars["abs_move"] = np.abs(bars["imbalance_qty_l1"].to_numpy())
        t = resolve_targets(bars, _ctx(), {"target_col": "abs_move"})[0]
        assert t.signed is False, "a non-negative target carries no direction"

    def test_binary_01_label_is_unsigned(self):
        from processes.targets import resolve_targets
        bars = _bars()
        bars["up_flag"] = (bars["imbalance_qty_l1"] > 0).astype(float)
        assert resolve_targets(bars, _ctx(), {"target_col": "up_flag"})[0].signed is False

    def test_unsigned_target_cannot_yield_a_proc1_polarity(self):
        """PROC-1 refuses findings without polarity; PROC-17 says where polarity is even
        definable. An unsigned target must not be able to claim one."""
        from processes.targets import resolve_targets
        bars = _bars()
        bars["abs_move"] = np.abs(bars["imbalance_qty_l1"].to_numpy())
        t = resolve_targets(bars, _ctx(), {"target_col": "abs_move"})[0]
        with pytest.raises(ValueError, match="unsigned"):
            t.polarity_of(bars["imbalance_qty_l1"].to_numpy())

    @pytest.mark.parametrize("sign", [1, -1])
    def test_polarity_recovers_a_planted_direction(self, sign):
        from processes.targets import resolve_targets
        n = 400
        rng = np.random.default_rng(5)
        f = rng.normal(size=n)
        # r(t) = p(t+1)/p(t) - 1 is driven by the increment at t+1, so the increment
        # array must be f shifted forward by one for f(t) to predict r(t).
        increments = np.r_[0.0, sign * 0.02 * f[:-1]]
        bars = pd.DataFrame({"raw_midprice": 100 + np.cumsum(increments),
                             "imbalance_qty_l1": f})
        t = resolve_targets(bars, _ctx(horizons={"h": 1}), {})[0]
        assert t.polarity_of(f, bars) == sign


# ── provenance: a target must describe itself for the surface / PROC-1 ───────────
class TestProvenance:
    def test_label_def_matches_surface_convention(self):
        from processes.targets import resolve_targets
        assert resolve_targets(_bars(), _ctx(), {"target_col": "tb_label"})[0].label_def == "tb_label"
        assert resolve_targets(_bars(), _ctx(), {})[0].label_def == "fwd_ret"

    def test_as_dict_is_json_serialisable(self):
        import json
        from processes.targets import resolve_targets
        for t in resolve_targets(_bars(), _ctx(), {"target_col": "tb_label"}):
            json.dumps(t.as_dict())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
