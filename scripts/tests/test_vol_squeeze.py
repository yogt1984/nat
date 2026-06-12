"""Tests for the vol_squeeze algorithm (A3)."""

import numpy as np
import pandas as pd
import pytest

from algorithms.registry import get_algorithm, list_algorithms
from algorithms.vol_squeeze import VolSqueeze


def _df(ratio, mid):
    return pd.DataFrame(
        {
            "vol_ratio_short_long": np.asarray(ratio, dtype=float),
            "raw_midprice": np.asarray(mid, dtype=float),
        }
    )


def _squeeze_then_breakout(direction=+1, seed=2):
    """Mixed-ratio history, deep sustained squeeze, expansion with a
    directional price move during the lookback window."""
    rng = np.random.default_rng(seed)
    n_hist, n_squeeze, n_post = 3000, 800, 500
    ratio = np.concatenate(
        [
            rng.uniform(0.3, 1.5, n_hist),     # populate percentiles
            np.full(n_squeeze, 0.10),          # deep squeeze
            np.full(n_post, 1.80),             # expansion
        ]
    )
    mid = np.full(len(ratio), 50_000.0)
    # price trends in `direction` through the squeeze tail + expansion
    ramp = np.linspace(0, direction * 50.0, n_squeeze + n_post)
    mid[n_hist:] += ramp
    return _df(ratio, mid), n_hist, n_squeeze


class TestContract:
    def test_registered(self):
        assert "vol_squeeze" in list_algorithms()
        algo = get_algorithm("vol_squeeze")
        assert algo.name() == "vol_squeeze"

    def test_step_returns_exact_feature_keys(self):
        algo = VolSqueeze()
        out = algo.step({"vol_ratio_short_long": 0.8, "raw_midprice": 100.0})
        assert set(out) == set(algo.feature_names)

    def test_nan_input_gives_nan_output(self):
        algo = VolSqueeze()
        for tick in (
            {"vol_ratio_short_long": np.nan, "raw_midprice": 100.0},
            {"vol_ratio_short_long": 0.8, "raw_midprice": np.nan},
            {},
        ):
            out = algo.step(tick)
            assert all(np.isnan(v) for v in out.values())

    def test_reset_reproduces_fresh_run(self):
        df, *_ = _squeeze_then_breakout()
        algo = VolSqueeze()
        first = algo.run_batch(df)
        second = algo.run_batch(df)  # run_batch resets internally
        pd.testing.assert_frame_equal(first, second)


class TestBehavior:
    def test_breakout_up_fires_positive(self):
        df, n_hist, n_squeeze = _squeeze_then_breakout(direction=+1)
        out = VolSqueeze().run_batch(df)
        sig = out["alg_vsq_breakout_signal"]
        # no signal during history or squeeze...
        assert (sig.iloc[:n_hist + n_squeeze].fillna(0) == 0).all()
        # ...fires positive shortly after the expansion begins
        post = sig.iloc[n_hist + n_squeeze:]
        assert post.max() > 0.9
        assert (post.dropna() >= 0).all()

    def test_breakout_down_fires_negative(self):
        df, n_hist, n_squeeze = _squeeze_then_breakout(direction=-1)
        out = VolSqueeze().run_batch(df)
        post = out["alg_vsq_breakout_signal"].iloc[n_hist + n_squeeze:]
        assert post.min() < -0.9
        assert (post.dropna() <= 0).all()

    def test_signal_decays_linearly(self):
        df, n_hist, n_squeeze = _squeeze_then_breakout()
        out = VolSqueeze(hold_ticks=200).run_batch(df)
        sig = out["alg_vsq_breakout_signal"].iloc[n_hist + n_squeeze:].to_numpy()
        peak = np.nanargmax(sig)
        tail = sig[peak: peak + 200]
        assert (np.diff(tail[np.isfinite(tail)]) <= 1e-12).all()

    def test_no_breakout_without_squeeze(self):
        rng = np.random.default_rng(4)
        n = 4000
        ratio = np.concatenate(
            [rng.uniform(0.5, 1.2, n), np.full(300, 1.9)]  # spike, no squeeze
        )
        mid = np.linspace(50_000, 50_100, len(ratio))
        out = VolSqueeze().run_batch(_df(ratio, mid))
        assert (out["alg_vsq_breakout_signal"].fillna(0) == 0).all()

    def test_squeeze_on_flag_during_squeeze(self):
        df, n_hist, n_squeeze = _squeeze_then_breakout()
        out = VolSqueeze().run_batch(df)
        armed_zone = out["alg_vsq_squeeze_on"].iloc[
            n_hist + 700: n_hist + n_squeeze
        ]
        assert (armed_zone == 1.0).all()

    def test_stale_squeeze_does_not_fire(self):
        """A squeeze that fades through the mid-band for long enough must
        not trigger a breakout on a much later expansion."""
        rng = np.random.default_rng(6)
        n_hist, n_squeeze, n_drift = 3000, 800, 2000
        ratio = np.concatenate(
            [
                rng.uniform(0.3, 1.5, n_hist),
                np.full(n_squeeze, 0.10),
                rng.uniform(0.7, 0.95, n_drift),  # mid-band drift, stale-out
                np.full(300, 1.85),               # late expansion
            ]
        )
        mid = np.linspace(50_000, 50_200, len(ratio))
        out = VolSqueeze(min_squeeze_ticks=600).run_batch(_df(ratio, mid))
        late = out["alg_vsq_breakout_signal"].iloc[n_hist + n_squeeze + n_drift:]
        assert (late.fillna(0) == 0).all()


class TestStepBatchParity:
    def test_step_equals_run_batch(self):
        rng = np.random.default_rng(9)
        n = 5000
        ratio = rng.uniform(0.1, 2.0, n)
        ratio[rng.random(n) < 0.02] = np.nan  # sprinkle NaNs
        mid = 50_000 + np.cumsum(rng.normal(0, 5, n))
        df = _df(ratio, mid)

        batch = VolSqueeze().run_batch(df)

        algo = VolSqueeze()
        rows = []
        for i in range(n):
            rows.append(
                algo.step(
                    {
                        "vol_ratio_short_long": ratio[i],
                        "raw_midprice": mid[i],
                    }
                )
            )
        stepped = pd.DataFrame(rows)
        stepped.iloc[: algo.warmup] = np.nan  # match warmup blanking

        pd.testing.assert_frame_equal(
            batch.reset_index(drop=True), stepped, check_exact=True
        )
