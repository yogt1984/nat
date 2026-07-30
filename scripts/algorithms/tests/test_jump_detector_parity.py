"""BUG-5 planted test — jump_detector (v1) exact step/batch parity.

run_batch()'s rolling bipower window ENDED at the current tick, so |r_t| entered its
own volatility denominator — self-masking: the statistic is understated at exactly the
jumps it exists to detect, and backtest ≠ live (step() correctly excludes the current
return via buffer[:-1]). The generic TestStepBatchConsistency only checks correlation
>0.9 and skips near-constant columns, so this was invisible. Confirmed by the Q4 kill
gate (FINDINGS §4.6, BUG-5). Mirrors v2's rtol=1e-9 parity test.

Contract:
  (a) at a planted jump, batch L equals step L (no self-masked understatement);
  (b) full tick-for-tick parity on all four outputs at rtol=1e-9;
  (c) identical detection sets (no threshold flips between backtest and live).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algorithms.jump_detector import JumpDetector

JUMPS = {500: 12.0, 1500: -15.0, 2500: 10.0, 3500: -12.0}


def _diffusion_with_jumps(n: int = 4500, sigma: float = 1e-5, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = sigma * rng.standard_normal(n)
    for idx, k in JUMPS.items():
        rets[idx] = k * sigma
    return 100.0 * np.exp(np.cumsum(rets))


def _run_both(prices: np.ndarray):
    df = pd.DataFrame({"raw_midprice": prices})
    batch = JumpDetector().run_batch(df)
    inst = JumpDetector()
    step = pd.DataFrame([inst.step({"raw_midprice": float(p)}) for p in prices])
    return batch, step


class TestExactParity:
    def test_all_outputs_match_step(self):
        prices = _diffusion_with_jumps()
        batch, step = _run_both(prices)
        warmup = JumpDetector().warmup
        for col in ("alg_jump_statistic", "alg_jump_detected",
                    "alg_jump_magnitude", "alg_post_jump_reversion"):
            b = batch[col].to_numpy()[warmup:]
            s = step[col].to_numpy()[warmup:]
            mask = np.isfinite(b) & np.isfinite(s)
            assert mask.sum() > 1000, col
            np.testing.assert_allclose(
                b[mask], s[mask], rtol=1e-9, atol=1e-15,
                err_msg=f"BUG-5 self-masking divergence in {col}",
            )

    def test_no_understatement_at_planted_jumps(self):
        prices = _diffusion_with_jumps()
        batch, step = _run_both(prices)
        for idx in JUMPS:
            b = batch["alg_jump_statistic"].iloc[idx]
            s = step["alg_jump_statistic"].iloc[idx]
            assert np.isfinite(b) and np.isfinite(s)
            # the old batch path understated L at jumps (current |r| in the denominator)
            assert b == pytest.approx(s, rel=1e-9), (
                f"jump @{idx}: batch L={b:.3f} vs step L={s:.3f}"
            )

    def test_identical_detection_sets(self):
        prices = _diffusion_with_jumps()
        batch, step = _run_both(prices)
        warmup = JumpDetector().warmup
        b = batch["alg_jump_detected"].to_numpy()[warmup:]
        s = step["alg_jump_detected"].to_numpy()[warmup:]
        mask = np.isfinite(b) & np.isfinite(s)
        flips = int(np.sum(b[mask] != s[mask]))
        assert flips == 0, f"{flips} backtest-vs-live detection flips"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
