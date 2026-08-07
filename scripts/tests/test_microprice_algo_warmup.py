"""HF1 `microprice` — the warmup contract, and the trap inside it.

`contracts/algorithm.md` requires that the first `warmup` rows of `run_batch()` come back
NaN: before the EMA has seen enough ticks its output is an artifact of its initialisation,
not a measurement, and downstream code cannot tell the difference. `MicrostructureAlgorithm.
run_batch` enforces this for every algorithm that uses the default path (`base.py:108-111`).
`microprice` overrides `run_batch` with a vectorised path and did **not** — caught by
`test_algorithm_smoke` on 2026-08-07 as "first 50 (warmup) rows not NaN-blanked".

The obvious fix is wrong in an interesting way, so both properties are pinned here:

  1. **The warmup rows must be NaN** — the contract itself.
  2. **Blanking must erase OUTPUT, never skip COMPUTATION.** The EMA is a recurrence: if
     you avoid the first 50 ticks instead of hiding their results, the state at tick 50 is
     a fresh EMA rather than one that has already absorbed 50 observations, and *every
     subsequent value changes*. A "fix" that blanks by starting late silently alters the
     whole series — including the HF1 centre used in the §4.7–§4.9 maker experiments.
     Test 2 is what distinguishes the two implementations.
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

from algorithms.microprice import Microprice  # noqa: E402


def _frame(n=400, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "imbalance_qty_l1": rng.uniform(0.0, 1.0, n),
        "raw_spread_bps": rng.uniform(0.5, 2.0, n),
        "raw_midprice": 50_000 + np.cumsum(rng.normal(scale=0.5, size=n)),
    })


def test_warmup_rows_are_nan_blanked():
    """The contract `test_algorithm_smoke` enforces for every algorithm."""
    algo = Microprice(warmup=50)
    out = algo.run_batch(_frame(400))

    w = algo.warmup
    assert w == 50
    assert out.iloc[:w].isna().all().all(), "warmup rows must be NaN"
    assert out.iloc[w:].notna().any().any(), "blanking must not erase the whole frame"


def test_post_warmup_values_are_unchanged_by_blanking():
    """The decisive one: blanking hides output, it must not restart the recurrence.

    Compare against the EMA computed over the *full* series. If the implementation
    blanked by skipping the first 50 ticks, the state at tick 50 would be fresh and every
    later value would differ.
    """
    df = _frame(400)
    algo = Microprice(warmup=50)

    # Reference: replay the recurrence tick-by-tick over the whole frame, no blanking.
    algo.reset()
    ref = []
    for _, row in df.iterrows():
        ref.append(algo.step({
            "imbalance_qty_l1": float(row["imbalance_qty_l1"]),
            "raw_spread_bps": float(row["raw_spread_bps"]),
            "raw_midprice": float(row["raw_midprice"]),
        })["alg_mp_dev_ema"])
    ref = np.asarray(ref, dtype=float)

    out = algo.run_batch(df)
    got = out["alg_mp_dev_ema"].to_numpy(dtype=float)

    w = algo.warmup
    np.testing.assert_allclose(got[w:], ref[w:], rtol=1e-12, atol=1e-12,
                               err_msg="post-warmup values changed — the recurrence was "
                                       "restarted instead of the output being hidden")


def test_short_frame_shorter_than_warmup_is_not_wholly_blanked():
    """`base.py` only blanks when `warmup < n`; the override must match that rule.

    A frame shorter than the warmup returns unblanked values rather than an all-NaN
    frame — same contract as every other algorithm, so callers see one behaviour.
    """
    algo = Microprice(warmup=50)
    out = algo.run_batch(_frame(20))
    assert len(out) == 20
    assert out.notna().any().any()


def test_zero_warmup_blanks_nothing():
    algo = Microprice(warmup=0)
    out = algo.run_batch(_frame(100))
    assert out["alg_mp_dev_bps"].notna().all()


def test_nan_inputs_still_propagate_after_the_fix():
    """Blanking must not paper over NaN handling: a NaN tick stays NaN, and the state freezes."""
    df = _frame(200)
    df.loc[120, "imbalance_qty_l1"] = np.nan

    out = Microprice(warmup=50).run_batch(df)
    assert out.loc[120].isna().all(), "a NaN input row must yield NaN outputs"
    assert out.loc[121:].notna().any().any(), "state must resume after a NaN tick"
