"""
Change-Point Detector — CUSUM + Bayesian Online Change-Point Detection
=======================================================================

Detects regime transitions using two complementary methods:

  1. CUSUM (Page 1954): bilateral cumulative sum on standardized imbalance.
     Detects sustained mean shifts. O(1) per bar, zero memory growth.

  2. Bayesian OCD (Adams & MacKay 2007): run-length distribution over
     imbalance under a Normal-Inverse-Gamma conjugate prior. Detects both
     mean and variance changes. O(max_run_length) per bar.

The composite signal combines CUSUM direction with Bayesian change probability.

Output Features (4):
  alg_cpd_cusum_signal   [-inf, inf]  CUSUM composite: max(S+, |S-|) * sign
  alg_cpd_run_length     [0, inf)     Expected bars since last change-point
  alg_cpd_change_prob    [0, 1]       P(change-point at current bar)
  alg_cpd_regime_age     [0, inf)     Bars since last CUSUM alarm

References:
  Page (1954) — Continuous Inspection Schemes, Biometrika 41.
  Adams & MacKay (2007) — Bayesian Online Changepoint Detection, arXiv:0710.3742.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class ChangePointDetector(MicrostructureAlgorithm):
    """CUSUM + Bayesian OCD for regime transition detection."""

    bar_level = True

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        cusum_drift: float = 0.05,
        hazard_rate: float = 0.005,
        max_run_length: int = 500,
        calibration_window: int = 200,
    ):
        self._cusum_threshold = cusum_threshold
        self._cusum_drift = cusum_drift
        self._hazard = hazard_rate
        self._max_rl = max_run_length
        self._cal_window = calibration_window

        # NIG prior parameters
        self._mu0 = 0.0
        self._kappa0 = 1.0
        self._alpha0 = 1.0
        self._beta0 = 1.0

        self._reset_state()

    def _reset_state(self) -> None:
        # CUSUM state
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._regime_age = 0

        # Calibration buffer for CUSUM normalization
        self._cal_buf: list[float] = []
        self._cal_mean = 0.0
        self._cal_std = 1.0

        # Bayesian OCD state — run-length distribution
        self._rl_probs = np.array([1.0])
        # NIG sufficient statistics per run length
        self._mu_n = np.array([self._mu0])
        self._kappa_n = np.array([self._kappa0])
        self._alpha_n = np.array([self._alpha0])
        self._beta_n = np.array([self._beta0])

        self._bar_count = 0

    def name(self) -> str:
        return "change_point_detector"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_cpd_cusum_signal", warmup=100,
                             description="CUSUM composite: max(S+,|S-|) * sign"),
            AlgorithmFeature("alg_cpd_run_length", warmup=100,
                             description="Expected run length (bars since change)"),
            AlgorithmFeature("alg_cpd_change_prob", warmup=100,
                             description="P(change-point at current bar)"),
            AlgorithmFeature("alg_cpd_regime_age", warmup=0,
                             description="Bars since last CUSUM alarm"),
        ]

    def required_columns(self) -> list[str]:
        return [
            "imbalance_qty_l1_mean",
            "vol_returns_5m_last",
            "ent_tick_1m_mean",
        ]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan_out = {f.name: np.nan for f in self.alg_features()}

        imb = tick.get("imbalance_qty_l1_mean", np.nan)
        if not np.isfinite(imb):
            return nan_out

        self._bar_count += 1
        self._regime_age += 1

        # --- Calibration: rolling mean/std for CUSUM normalization ---
        self._cal_buf.append(imb)
        if len(self._cal_buf) > self._cal_window:
            self._cal_buf.pop(0)

        if len(self._cal_buf) >= 20:
            arr = np.array(self._cal_buf)
            self._cal_mean = float(np.mean(arr))
            self._cal_std = max(float(np.std(arr)), 1e-10)

        x = (imb - self._cal_mean) / self._cal_std

        # --- CUSUM update ---
        self._cusum_pos = max(0.0, self._cusum_pos + x - self._cusum_drift)
        self._cusum_neg = max(0.0, self._cusum_neg - x - self._cusum_drift)

        # Check for alarm
        if self._cusum_pos > self._cusum_threshold:
            self._cusum_pos = 0.0
            self._regime_age = 0
        if self._cusum_neg > self._cusum_threshold:
            self._cusum_neg = 0.0
            self._regime_age = 0

        # Composite signal: magnitude * direction
        if self._cusum_pos >= self._cusum_neg:
            cusum_signal = self._cusum_pos
        else:
            cusum_signal = -self._cusum_neg

        # --- Bayesian OCD update (Adams & MacKay 2007) ---
        change_prob, expected_rl = self._bayesian_update(imb)

        return {
            "alg_cpd_cusum_signal": cusum_signal,
            "alg_cpd_run_length": expected_rl,
            "alg_cpd_change_prob": change_prob,
            "alg_cpd_regime_age": float(self._regime_age),
        }

    def _bayesian_update(self, x: float) -> tuple[float, float]:
        """Adams-MacKay Bayesian online change-point detection.

        Returns (change_probability, expected_run_length).
        """
        n = len(self._rl_probs)
        H = self._hazard

        # Step 1: Predictive probabilities under Student-t for each run length
        pred_probs = np.empty(n)
        for r in range(n):
            mu = self._mu_n[r]
            kappa = self._kappa_n[r]
            alpha = self._alpha_n[r]
            beta = self._beta_n[r]

            # Student-t predictive: t_{2*alpha}(mu, beta*(kappa+1)/(alpha*kappa))
            df = 2.0 * alpha
            scale2 = beta * (kappa + 1.0) / (alpha * kappa)
            scale = np.sqrt(max(scale2, 1e-20))

            if df > 0 and scale > 0:
                pred_probs[r] = student_t.pdf(x, df, loc=mu, scale=scale)
            else:
                pred_probs[r] = 1e-10

        # Step 2: Growth probabilities (run length increases)
        growth = self._rl_probs * (1.0 - H) * pred_probs

        # Step 3: Change-point probability (run length resets to 0)
        # Prior predictive for new run
        df0 = 2.0 * self._alpha0
        scale0 = np.sqrt(max(self._beta0 * (self._kappa0 + 1.0) / (self._alpha0 * self._kappa0), 1e-20))
        prior_pred = student_t.pdf(x, df0, loc=self._mu0, scale=scale0)
        cp_prob = np.sum(self._rl_probs * H * pred_probs) + 1e-300

        # Step 4: New joint distribution
        new_joint = np.empty(n + 1)
        new_joint[0] = cp_prob
        new_joint[1:] = growth

        # Normalize
        evidence = np.sum(new_joint)
        if evidence > 0:
            new_joint /= evidence
        else:
            new_joint = np.ones(n + 1) / (n + 1)

        # Step 5: Update sufficient statistics
        # For existing run lengths: update NIG parameters
        new_mu = np.empty(n + 1)
        new_kappa = np.empty(n + 1)
        new_alpha = np.empty(n + 1)
        new_beta = np.empty(n + 1)

        # New run (r=0): reset to prior
        new_mu[0] = self._mu0
        new_kappa[0] = self._kappa0
        new_alpha[0] = self._alpha0
        new_beta[0] = self._beta0

        # Existing runs: NIG posterior update
        old_mu = self._mu_n
        old_kappa = self._kappa_n
        new_kappa[1:] = old_kappa + 1.0
        new_mu[1:] = (old_kappa * old_mu + x) / new_kappa[1:]
        new_alpha[1:] = self._alpha_n + 0.5
        new_beta[1:] = self._beta_n + old_kappa * (x - old_mu) ** 2 / (2.0 * new_kappa[1:])

        # Truncate at max_run_length
        if len(new_joint) > self._max_rl:
            new_joint = new_joint[:self._max_rl]
            new_mu = new_mu[:self._max_rl]
            new_kappa = new_kappa[:self._max_rl]
            new_alpha = new_alpha[:self._max_rl]
            new_beta = new_beta[:self._max_rl]
            # Renormalize after truncation
            s = np.sum(new_joint)
            if s > 0:
                new_joint /= s

        self._rl_probs = new_joint
        self._mu_n = new_mu
        self._kappa_n = new_kappa
        self._alpha_n = new_alpha
        self._beta_n = new_beta

        # Outputs
        change_prob = float(new_joint[0])
        r_vals = np.arange(len(new_joint))
        expected_rl = float(np.sum(r_vals * new_joint))

        return change_prob, expected_rl

    def reset(self) -> None:
        self._reset_state()
