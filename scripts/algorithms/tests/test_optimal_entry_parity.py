"""BUG-4 planted test — optimal_entry backtest/live parity on sigma_process.

run_batch() rebuilt its Kalman filter with a hardcoded sigma_process=0.01 (the
configured value was never stored on the instance), so every backtest was blind to
the parameter while the live step() path honored it — a backtest/live parity break
documented since 2026-06-12 and confirmed by the Q4 kill gate (FINDINGS §4.6, BUG-4).

Contract:
  (a) run_batch RESPONDS to sigma_process (a 50x change must alter the output);
  (b) step() and run_batch() agree tick-for-tick at a NON-default sigma_process
      (the exact configuration a live deployment would run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algorithms.optimal_entry import OptimalEntry


def _planted_df(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """OU-ish imbalance series with a drifting stretch so the SPRT actually fires."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        drift = 0.15 if 1000 <= i < 1400 else 0.0        # planted evidence stretch
        x[i] = 0.9 * x[i - 1] + drift + 0.1 * rng.standard_normal()
    return pd.DataFrame({"imbalance_qty_l1": x})


class TestSigmaProcessThreaded:
    def test_run_batch_responds_to_sigma_process(self):
        df = _planted_df()
        lo = OptimalEntry(sigma_process=0.01).run_batch(df)
        hi = OptimalEntry(sigma_process=0.5).run_batch(df)
        stat_lo = lo["alg_sprt_statistic"].to_numpy()
        stat_hi = hi["alg_sprt_statistic"].to_numpy()
        mask = np.isfinite(stat_lo) & np.isfinite(stat_hi)
        assert mask.sum() > 100
        assert not np.allclose(stat_lo[mask], stat_hi[mask]), (
            "BUG-4: run_batch is blind to sigma_process — the backtest never "
            "exercises the configured filter"
        )


class TestStepBatchParity:
    @pytest.mark.parametrize("sigma_process", [0.005, 0.05, 0.5])
    def test_parity_at_nondefault_sigma(self, sigma_process):
        df = _planted_df(seed=1)
        batch = OptimalEntry(sigma_process=sigma_process).run_batch(df)

        inst = OptimalEntry(sigma_process=sigma_process)
        step = pd.DataFrame(
            [inst.step({"imbalance_qty_l1": float(v)}) for v in df["imbalance_qty_l1"]]
        )

        warmup = OptimalEntry().warmup
        for col in ("alg_sprt_statistic", "alg_entry_signal", "alg_cumulative_evidence"):
            b = batch[col].to_numpy()[warmup:]
            s = step[col].to_numpy()[warmup:]
            mask = np.isfinite(b) & np.isfinite(s)
            assert mask.sum() > 100, col
            np.testing.assert_allclose(
                b[mask], s[mask], rtol=1e-9, atol=1e-12,
                err_msg=f"step/batch divergence in {col} at sigma_process={sigma_process}",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
