"""Tests for cross-algorithm ensemble layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from algorithms.ensemble import Ensemble, DEFAULT_SIGNAL_FEATURES  # noqa: E402


# --- Helpers ---

def _mock_results(n: int = 500, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create mock algorithm results with realistic signal shapes."""
    rng = np.random.default_rng(seed)
    return {
        "jump_detector": pd.DataFrame({
            "alg_jump_statistic": rng.normal(0, 1, n),
            "alg_jump_detected": (rng.random(n) > 0.95).astype(float),
            "alg_jump_magnitude": rng.normal(0, 0.01, n),
            "alg_post_jump_reversion": rng.normal(0, 0.5, n),
        }),
        "optimal_entry": pd.DataFrame({
            "alg_sprt_statistic": rng.normal(0, 1, n),
            "alg_entry_signal": rng.choice([-1.0, 0.0, 1.0], n, p=[0.1, 0.8, 0.1]),
            "alg_cumulative_evidence": rng.uniform(0, 1, n),
        }),
        "funding_reversion": pd.DataFrame({
            "alg_funding_signal": rng.choice([-1.0, 0.0, 1.0], n, p=[0.15, 0.7, 0.15]),
            "alg_funding_momentum": rng.normal(0, 0.001, n),
            "alg_premium_divergence": rng.normal(0, 0.5, n),
            "alg_funding_halflife_ticks": rng.uniform(100, 500, n),
        }),
        "surprise_signal": pd.DataFrame({
            "alg_entropy_surprise": rng.normal(0, 1, n),
            "alg_entropy_roc": rng.normal(0, 0.1, n),
            "alg_regime_transition_prob": rng.uniform(0, 1, n),
        }),
        "weighted_ofi": pd.DataFrame({
            "alg_weighted_ofi": rng.normal(0, 100, n),
            "alg_ofi_momentum": rng.normal(0, 50, n),
            "alg_ofi_divergence": rng.normal(0, 30, n),
        }),
    }


def _mock_base_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "raw_midprice": 67000 + np.cumsum(rng.normal(0, 1, n)),
        "ent_book_shape": rng.uniform(0, 1, n),
    })


def _mock_forward_returns(n: int = 500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.001, n)


# --- equal_weight ---

class TestEqualWeight:

    def test_produces_ens_signal(self):
        results = _mock_results()
        ens = Ensemble(["jump_detector", "optimal_entry"], method="equal_weight")
        out = ens.combine(results)
        assert "ens_signal" in out.columns
        assert len(out) == 500

    def test_weights_are_equal(self):
        results = _mock_results()
        algos = ["jump_detector", "optimal_entry", "funding_reversion"]
        ens = Ensemble(algos, method="equal_weight")
        out = ens.combine(results)
        for name in algos:
            col = f"ens_weight_{name}"
            assert col in out.columns
            assert np.allclose(out[col].values, 1.0 / 3)

    def test_signal_is_mean_of_zscored(self):
        results = _mock_results(n=200)
        algos = ["jump_detector", "weighted_ofi"]
        ens = Ensemble(algos, method="equal_weight")
        out = ens.combine(results)
        # ens_signal should be finite and have zero mean (z-scored inputs)
        sig = out["ens_signal"].values
        assert np.all(np.isfinite(sig))
        assert abs(np.mean(sig)) < 0.1  # approximately zero

    def test_subset_of_algorithms(self):
        results = _mock_results()
        ens = Ensemble(["weighted_ofi"], method="equal_weight")
        out = ens.combine(results)
        assert "ens_signal" in out.columns
        assert "ens_weight_weighted_ofi" in out.columns

    def test_missing_algorithm_warns(self, caplog):
        import logging
        results = _mock_results()
        ens = Ensemble(["jump_detector", "nonexistent"], method="equal_weight")
        with caplog.at_level(logging.WARNING, logger="algorithms.ensemble"):
            out = ens.combine(results)
        assert any("nonexistent" in r.message for r in caplog.records)

    def test_all_five_winners(self):
        results = _mock_results()
        ens = Ensemble(list(DEFAULT_SIGNAL_FEATURES.keys()), method="equal_weight")
        out = ens.combine(results)
        assert len(out.columns) == 1 + 5  # ens_signal + 5 weights


# --- ic_weight ---

class TestICWeight:

    def test_produces_weighted_signal(self):
        n = 6000
        results = _mock_results(n=n)
        fwd = _mock_forward_returns(n=n)
        algos = ["jump_detector", "weighted_ofi"]
        ens = Ensemble(algos, method="ic_weight", ic_lookback=1000)
        out = ens.combine(results, forward_returns=fwd)
        assert "ens_signal" in out.columns
        assert "ens_weight_jump_detector" in out.columns
        assert "ens_weight_weighted_ofi" in out.columns

    def test_weights_sum_to_one(self):
        n = 6000
        results = _mock_results(n=n)
        fwd = _mock_forward_returns(n=n)
        algos = ["jump_detector", "weighted_ofi"]
        ens = Ensemble(algos, method="ic_weight", ic_lookback=1000)
        out = ens.combine(results, forward_returns=fwd)
        # After lookback, weights should sum to ~1
        weight_cols = [c for c in out.columns if c.startswith("ens_weight_")]
        weights_sum = out[weight_cols].sum(axis=1)
        valid = weights_sum[weights_sum > 0]
        assert np.allclose(valid, 1.0, atol=1e-6)

    def test_requires_forward_returns(self):
        results = _mock_results()
        ens = Ensemble(["jump_detector"], method="ic_weight")
        with pytest.raises(ValueError, match="forward_returns"):
            ens.combine(results)


# --- regime_switch ---

class TestRegimeSwitch:

    def test_produces_signal(self):
        n = 500
        results = _mock_results(n=n)
        base = _mock_base_df(n=n)
        fwd = _mock_forward_returns(n=n)
        algos = ["jump_detector", "funding_reversion", "weighted_ofi"]
        ens = Ensemble(algos, method="regime_switch")
        out = ens.combine(results, base_df=base, forward_returns=fwd)
        assert "ens_signal" in out.columns
        assert len(out) == n

    def test_trending_vs_reverting_weights(self):
        n = 500
        results = _mock_results(n=n)
        base = _mock_base_df(n=n)
        fwd = _mock_forward_returns(n=n)
        algos = ["jump_detector", "funding_reversion"]
        ens = Ensemble(algos, method="regime_switch")
        out = ens.combine(results, base_df=base, forward_returns=fwd)
        # jump_detector is trending, funding_reversion is reverting
        # In low-entropy regime, jump_detector should have higher weight
        w_jump = out["ens_weight_jump_detector"].values
        w_fund = out["ens_weight_funding_reversion"].values
        # At least some ticks should have non-zero jump_detector weight
        assert (w_jump > 0).sum() > 100
        assert (w_fund > 0).sum() > 100

    def test_requires_base_df(self):
        results = _mock_results()
        fwd = _mock_forward_returns()
        ens = Ensemble(["jump_detector"], method="regime_switch")
        with pytest.raises(ValueError, match="base_df"):
            ens.combine(results, forward_returns=fwd)

    def test_fallback_when_regime_missing(self, caplog):
        import logging
        n = 500
        results = _mock_results(n=n)
        base = pd.DataFrame({"raw_midprice": np.ones(n)})  # no regime col
        fwd = _mock_forward_returns(n=n)
        ens = Ensemble(["jump_detector", "weighted_ofi"], method="regime_switch")
        with caplog.at_level(logging.WARNING, logger="algorithms.ensemble"):
            out = ens.combine(results, base_df=base, forward_returns=fwd)
        assert "ens_signal" in out.columns  # falls back to equal_weight


# --- Custom signal features ---

class TestCustomSignals:

    def test_custom_signal_feature(self):
        results = _mock_results()
        ens = Ensemble(
            ["jump_detector"],
            signal_features={"jump_detector": "alg_jump_statistic"},
        )
        out = ens.combine(results)
        assert "ens_signal" in out.columns

    def test_unknown_algorithm_without_default_raises(self):
        results = {"custom_algo": pd.DataFrame({"alg_custom": np.ones(100)})}
        ens = Ensemble(["custom_algo"])
        with pytest.raises(ValueError, match="No signal feature"):
            ens.combine(results)

    def test_unknown_algorithm_with_mapping_works(self):
        results = {"custom_algo": pd.DataFrame({"alg_custom": np.ones(100)})}
        ens = Ensemble(
            ["custom_algo"],
            signal_features={"custom_algo": "alg_custom"},
        )
        out = ens.combine(results)
        assert "ens_signal" in out.columns


# --- e2e with real fixture ---

FIXTURE = ROOT / "scripts" / "algorithms" / "tests" / "fixtures" / "btc_1h_real.parquet"


@pytest.mark.skipif(not FIXTURE.exists(), reason="Real data fixture not available")
class TestEnsembleOnRealData:

    def test_equal_weight_on_fixture(self):
        from algorithms.autodiscover import discover_all
        from algorithms.registry import get_algorithm
        from algorithms.runner import AlgorithmRunner

        discover_all()
        df = pd.read_parquet(FIXTURE)

        results = {}
        for name in ["jump_detector", "optimal_entry"]:
            alg = get_algorithm(name)
            runner = AlgorithmRunner(alg)
            res = runner.run_on_dataframe(df)
            results[name] = res.features_df

        ens = Ensemble(["jump_detector", "optimal_entry"], method="equal_weight")
        out = ens.combine(results)

        assert len(out) == len(df)
        assert out["ens_signal"].notna().mean() > 0.5
        assert np.all(np.isfinite(out["ens_signal"].dropna()))
