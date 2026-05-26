"""
BTC Price Probability Model

Estimates P(BTC > strike | current features) using NAT's 209-feature vector.
Two models:
  1. Analytical: Conditional log-normal with feature-derived σ and μ
  2. Empirical: Historical frequency of BTC crossing strike from current conditions

The key insight: NAT's vol/regime/entropy features give better σ and μ estimates
than the market, which prices binary outcomes using simple implied vol.

Usage:
    from polymarket.probability_model import ProbabilityModel
    model = ProbabilityModel()
    model.fit(feature_df)  # historical calibration
    p = model.predict(current_features, strike=108000, horizon_minutes=5)
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from typing import Optional


class ProbabilityModel:
    """
    Estimate P(BTC_T > K | features_t) at horizon T-t.

    Uses a conditional Gaussian model where drift and volatility are
    functions of the current feature state (regime, entropy, momentum).

    P(S_T > K) = Φ((log(S/K) + μτ) / (σ√τ))

    where μ and σ are conditioned on current features:
      σ = f(vol_garman_klass_5m, vol_returns_1m, regime_state)
      μ = f(trend_momentum_300, ctx_funding_zscore, ent_book_shape)
    """

    def __init__(self):
        # Calibrated parameters (set by fit())
        self._vol_scale: float = 1.0      # Scaling from feature-vol to actual vol
        self._drift_coefs: np.ndarray = np.zeros(4)  # Regression coefficients
        self._vol_coefs: np.ndarray = np.zeros(3)
        self._base_vol: float = 0.0       # Unconditional vol (per minute, in log-return)
        self._base_drift: float = 0.0     # Unconditional drift
        self._fitted: bool = False

        # Feature names used for conditioning
        self._vol_features = ["vol_parkinson_5m", "vol_returns_1m", "vol_zscore"]
        self._drift_features = [
            "trend_momentum_300", "ctx_funding_zscore",
            "ent_book_shape", "trend_hurst_300",
        ]

    def fit(self, df: "pd.DataFrame", horizon_minutes: int = 5) -> "ProbabilityModel":
        """
        Calibrate the conditional model on historical data.

        For each row, compute actual forward return at horizon,
        then regress realized vol and drift on features.
        """
        import pandas as pd

        horizon_rows = horizon_minutes * 60  # 1 row per second in aligned data
        if len(df) < horizon_rows * 2:
            raise ValueError(f"Need at least {horizon_rows * 2} rows, got {len(df)}")

        mid = df["raw_midprice"].values if "raw_midprice" in df.columns else df["BTC_raw_midprice"].values
        log_ret = np.log(mid[horizon_rows:] / mid[:-horizon_rows])

        # Trim to match
        n = len(log_ret)
        log_ret = log_ret[:n]

        # Unconditional stats
        self._base_vol = np.std(log_ret)
        self._base_drift = np.mean(log_ret)

        # Conditional volatility regression
        vol_X = self._extract_features(df, self._vol_features)[:n]
        mask = np.isfinite(vol_X).all(axis=1) & np.isfinite(log_ret)
        if mask.sum() < 1000:
            print(f"  Warning: only {mask.sum()} valid rows for vol regression")
            self._fitted = True
            return self

        # Target: realized abs(log_ret) as proxy for vol
        abs_ret = np.abs(log_ret[mask])
        X_vol = vol_X[mask]

        # Simple linear regression: abs_ret = a + b @ X_vol
        X_aug = np.column_stack([np.ones(len(X_vol)), X_vol])
        try:
            self._vol_coefs, _, _, _ = np.linalg.lstsq(X_aug, abs_ret, rcond=None)
        except np.linalg.LinAlgError:
            self._vol_coefs = np.zeros(X_aug.shape[1])
            self._vol_coefs[0] = np.mean(abs_ret)

        # Conditional drift regression
        drift_X = self._extract_features(df, self._drift_features)[:n]
        mask_d = np.isfinite(drift_X).all(axis=1) & np.isfinite(log_ret)
        if mask_d.sum() > 1000:
            X_drift = drift_X[mask_d]
            y_drift = log_ret[mask_d]
            X_aug_d = np.column_stack([np.ones(len(X_drift)), X_drift])
            try:
                self._drift_coefs, _, _, _ = np.linalg.lstsq(X_aug_d, y_drift, rcond=None)
            except np.linalg.LinAlgError:
                self._drift_coefs = np.zeros(X_aug_d.shape[1])

        self._fitted = True
        print(f"  Model fitted: base_vol={self._base_vol:.6f}/period, "
              f"base_drift={self._base_drift:.6f}/period, n={mask.sum()}")
        return self

    def predict(
        self,
        features: dict[str, float],
        current_price: float,
        strike: float,
        horizon_minutes: float,
    ) -> float:
        """
        Estimate P(BTC > strike at t + horizon | current features).

        Returns probability in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Conditional volatility
        vol_x = np.array([features.get(f, np.nan) for f in self._vol_features])
        if np.isfinite(vol_x).all():
            x_aug = np.concatenate([[1.0], vol_x])
            cond_vol = max(float(x_aug @ self._vol_coefs), self._base_vol * 0.3)
        else:
            cond_vol = self._base_vol

        # Scale vol to target horizon (vol fitted at calibration horizon)
        # If calibration was at 5min and we're predicting at H minutes:
        # σ_H = σ_5 * sqrt(H/5)
        sigma = cond_vol * np.sqrt(max(horizon_minutes / 5.0, 0.1))

        # Conditional drift
        drift_x = np.array([features.get(f, np.nan) for f in self._drift_features])
        if np.isfinite(drift_x).all():
            x_aug_d = np.concatenate([[1.0], drift_x])
            cond_drift = float(x_aug_d @ self._drift_coefs)
        else:
            cond_drift = self._base_drift

        mu = cond_drift * (horizon_minutes / 5.0)

        # P(S_T > K) = P(log(S_T/S_t) > log(K/S_t))
        # = P(Z > (log(K/S) - μ) / σ)  where Z ~ N(0,1)
        # = Φ((log(S/K) + μ) / σ)
        log_moneyness = np.log(current_price / strike)
        d = (log_moneyness + mu) / (sigma + 1e-12)
        prob = float(sp_stats.norm.cdf(d))

        return np.clip(prob, 0.001, 0.999)

    def predict_with_confidence(
        self,
        features: dict[str, float],
        current_price: float,
        strike: float,
        horizon_minutes: float,
    ) -> tuple[float, float]:
        """
        Returns (probability, confidence) where confidence reflects
        how much the conditional model differs from unconditional.

        High confidence = features strongly shift probability from base case.
        """
        prob = self.predict(features, current_price, strike, horizon_minutes)

        # Unconditional probability
        log_moneyness = np.log(current_price / strike)
        d_base = (log_moneyness + self._base_drift) / (self._base_vol + 1e-12)
        prob_base = float(sp_stats.norm.cdf(d_base))

        # Confidence: how much features move probability
        confidence = abs(prob - prob_base)
        return prob, confidence

    def _extract_features(self, df: "pd.DataFrame", feature_names: list[str]) -> np.ndarray:
        """Extract feature matrix, filling missing with NaN."""
        cols = []
        for f in feature_names:
            # Handle prefixed columns (BTC_vol_returns_1m etc.)
            if f in df.columns:
                cols.append(df[f].values)
            else:
                # Try with BTC_ prefix
                btc_f = f"BTC_{f}"
                if btc_f in df.columns:
                    cols.append(df[btc_f].values)
                else:
                    cols.append(np.full(len(df), np.nan))
        return np.column_stack(cols)


class EmpiricalModel:
    """
    Non-parametric probability estimation using historical feature bins.

    Bins current features into quantiles, looks up historical frequency
    of BTC crossing the strike from that feature state.

    More robust than Gaussian assumption, but needs more data.
    """

    def __init__(self, n_bins: int = 10):
        self._n_bins = n_bins
        self._bin_edges: dict[str, np.ndarray] = {}
        self._crossing_freq: Optional[np.ndarray] = None
        self._features_used: list[str] = [
            "vol_parkinson_5m", "trend_momentum_300", "ent_book_shape"
        ]

    def fit(self, df: "pd.DataFrame", horizon_minutes: int = 5) -> "EmpiricalModel":
        """
        Build lookup table: for each feature bin combination,
        what fraction of times did BTC move more than X bps?
        """
        import pandas as pd

        horizon_rows = horizon_minutes * 60
        mid = df["raw_midprice"].values if "raw_midprice" in df.columns else df["BTC_raw_midprice"].values
        fwd_ret = mid[horizon_rows:] / mid[:-horizon_rows] - 1
        n = len(fwd_ret)

        # Bin each feature
        bin_indices = []
        for f in self._features_used:
            col = f if f in df.columns else f"BTC_{f}"
            if col not in df.columns:
                continue
            vals = df[col].values[:n]
            edges = np.nanpercentile(vals, np.linspace(0, 100, self._n_bins + 1))
            self._bin_edges[f] = edges
            idx = np.searchsorted(edges[1:-1], vals)
            bin_indices.append(idx)

        if not bin_indices:
            raise ValueError("No valid features found")

        # Build multi-dimensional frequency table
        # Store: for each bin combo, distribution of forward returns
        self._return_distributions = {}
        combined_bins = np.column_stack(bin_indices) if len(bin_indices) > 1 else bin_indices[0].reshape(-1, 1)

        for i in range(n):
            if not np.isfinite(fwd_ret[i]):
                continue
            key = tuple(combined_bins[i])
            if key not in self._return_distributions:
                self._return_distributions[key] = []
            self._return_distributions[key].append(fwd_ret[i])

        print(f"  Empirical model: {len(self._return_distributions)} bins, "
              f"{sum(len(v) for v in self._return_distributions.values())} total obs")
        return self

    def predict(
        self,
        features: dict[str, float],
        current_price: float,
        strike: float,
        horizon_minutes: float,
    ) -> Optional[float]:
        """Estimate P(BTC > strike) from empirical distribution in feature bin."""
        required_return = strike / current_price - 1.0

        # Find bin for current features
        bin_key = []
        for f in self._features_used:
            if f not in self._bin_edges:
                continue
            val = features.get(f, features.get(f"BTC_{f}", np.nan))
            if not np.isfinite(val):
                return None
            edges = self._bin_edges[f]
            idx = int(np.searchsorted(edges[1:-1], val))
            bin_key.append(idx)

        key = tuple(bin_key)
        if key not in self._return_distributions:
            return None

        returns = np.array(self._return_distributions[key])
        if len(returns) < 30:
            return None

        prob = float(np.mean(returns > required_return))
        return np.clip(prob, 0.001, 0.999)
