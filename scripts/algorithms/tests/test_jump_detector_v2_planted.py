"""
Planted (synthetic) test-first contract for `jump_detector_v2` — RED before
implementation.

This file must FAIL until `scripts/algorithms/jump_detector_v2.py` exists,
defining `JumpDetectorV2(MicrostructureAlgorithm)` registered under the name
"jump_detector_v2". Per NAT methodology (docs/METHODOLOGY.md), the planted
test is written *before* the implementation so it encodes the contract
rather than the implementation's accidental behavior.

The frozen v1 baseline (`scripts/algorithms/jump_detector.py`,
`JumpDetector`) is NOT touched or re-tested here — it is out of scope.

Contract under test (jump_detector_v2 spec)
--------------------------------------------
JumpDetectorV2 upgrades the Lee-Mykland test with:
  1. An EVT (Gumbel) detection threshold instead of a flat significance
     constant, computable via `evt_threshold()`:
        c        = sqrt(2/pi)
        C_n      = sqrt(2 ln n)/c - (ln(pi) + ln(ln n)) / (2 c sqrt(2 ln n))
        S_n      = 1 / (c sqrt(2 ln n))
        beta*    = -ln(-ln(1 - alpha))
        threshold = C_n + S_n * beta*,  n = evt_block
  2. Staggered (skip-one) bipower variation |r_i|*|r_{i-2}| as an
     alternative to the adjacent-pair |r_i|*|r_{i-1}| estimator, to avoid a
     jump return directly polluting the very next tick's local-vol estimate.
  3. Directional asymmetry outputs (alg_jd2_rev_up / alg_jd2_rev_down) that
     route the (single) reversion signal by the sign of the triggering jump.
  4. A magnitude-adaptive reversion horizon: bigger detected jumps get a
     longer post-jump tracking window (floor = reversion_horizon, slope =
     horizon_per_l, cap = horizon_max).

Output contract — step() / run_batch() columns, in this exact order/set:
    alg_jd2_statistic, alg_jd2_detected, alg_jd2_magnitude,
    alg_jd2_reversion, alg_jd2_rev_up, alg_jd2_rev_down

Required input: ["raw_midprice"] only (same as v1).

Non-negotiable rule for this file: every assertion below targets the
*contract*, not a guessed implementation. Nothing here is weakened to make
red turn green — that is the implementer's job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Match the import convention used across scripts/algorithms/tests/
# (nat-research is installed editable from scripts/, so `algorithms` is a
# top-level importable package — see test_algorithms.py, test_real_data.py).
from algorithms.autodiscover import discover_all

discover_all()

from algorithms.registry import get_algorithm, list_algorithms  # noqa: E402

ALGO_NAME = "jump_detector_v2"

EXPECTED_KEYS = {
    "alg_jd2_statistic",
    "alg_jd2_detected",
    "alg_jd2_magnitude",
    "alg_jd2_reversion",
    "alg_jd2_rev_up",
    "alg_jd2_rev_down",
}


# ══════════════════════════════════════════════════════════════════════════
#  Synthetic planted-signal helpers (no real data anywhere in this file)
# ══════════════════════════════════════════════════════════════════════════

def _diffusion_with_jumps(n: int, sigma: float, jump_specs: dict[int, float],
                           seed: int, p0: float = 100.0) -> np.ndarray:
    """i.i.d. N(0, sigma^2) log-return diffusion of length n, with the
    log-return at each key of `jump_specs` REPLACED (not added to) by
    `multiple * sigma` — so the planted jump's true magnitude is known
    exactly, uncorrupted by the background draw at that tick.

    Returns the price path (prices[0] == p0).
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, n)
    rets[0] = 0.0
    for idx, mult in jump_specs.items():
        rets[idx] = mult * sigma
    return p0 * np.exp(np.cumsum(rets))


def _decaying_jump_series(n: int, jump_idx: int, jump_price: float,
                          decay_ticks: int, decay_step: float,
                          base_price: float = 100.0) -> np.ndarray:
    """Flat baseline (no diffusion noise) with a single isolated jump to
    `jump_price` at `jump_idx`, followed by a monotone partial decay back
    toward `base_price` over `decay_ticks` ticks (mirrors the frozen v1
    fixture in test_winning_algos.py::test_reversion_sign_convention, so the
    sign convention stays identical between v1 and v2).
    """
    prices = np.full(n, base_price)
    prices[jump_idx] = jump_price
    direction = np.sign(base_price - jump_price)  # which way decay moves
    for k in range(1, decay_ticks):
        prices[jump_idx + k] = jump_price + direction * decay_step * k
    return prices


# ══════════════════════════════════════════════════════════════════════════
#  1. Registry: the unit must exist and be discoverable under its own name
# ══════════════════════════════════════════════════════════════════════════

def test_registered_and_instantiable():
    """Plants: `jump_detector_v2` must be @register-ed and instantiable via
    the registry, exactly like every other algorithm (test_algorithms.py's
    parametrization sweeps `list_algorithms()`, so an unregistered unit is
    silently invisible to the whole test-all-algorithms suite).
    """
    assert ALGO_NAME in list_algorithms(), (
        f"'{ALGO_NAME}' is not registered — expected scripts/algorithms/"
        f"jump_detector_v2.py to define a @register-ed JumpDetectorV2"
    )
    inst = get_algorithm(ALGO_NAME)
    assert inst.name() == ALGO_NAME


def test_required_columns_is_raw_midprice_only():
    """Plants: the only input column is raw_midprice (no auxiliary features)."""
    inst = get_algorithm(ALGO_NAME)
    assert inst.required_columns() == ["raw_midprice"]


# ══════════════════════════════════════════════════════════════════════════
#  2. Contract conformance: exact output keys, NaN propagation, reset, prefix
# ══════════════════════════════════════════════════════════════════════════

def test_step_returns_exact_keys():
    """Plants: step() returns EXACTLY the 6 declared keys — no more, no
    fewer (base.py's ABC contract: 'step() must return exactly the keys
    from alg_features()')."""
    inst = get_algorithm(ALGO_NAME)
    out = inst.step({"raw_midprice": 100.0})
    assert set(out.keys()) == EXPECTED_KEYS


def test_alg_features_match_expected_names_and_prefix():
    """Plants: alg_features() names are exactly the 6 contract names, and
    every one starts with the mandatory 'alg_' prefix."""
    inst = get_algorithm(ALGO_NAME)
    names = [f.name for f in inst.alg_features()]
    assert set(names) == EXPECTED_KEYS
    assert len(names) == len(EXPECTED_KEYS), "duplicate feature names"
    assert all(n.startswith("alg_") for n in names)


def test_nan_input_yields_all_nan_output_mid_stream():
    """Plants: 'Handle NaN inputs gracefully: if any required column is
    NaN, return NaN for all outputs' — checked after the instance has
    already left warmup (i.e., mid-operation, not just during startup)."""
    inst = get_algorithm(ALGO_NAME)
    warm_prices = _diffusion_with_jumps(200, 1e-5, {}, seed=1)
    for p in warm_prices:
        inst.step({"raw_midprice": float(p)})

    out = inst.step({"raw_midprice": np.nan})
    assert set(out.keys()) == EXPECTED_KEYS
    assert all(np.isnan(v) for v in out.values()), f"expected all-NaN, got {out}"


def test_reset_restores_initial_state():
    """Plants: reset() clears all internal state, so replaying the *same*
    tick sequence after reset() reproduces identical output — required
    because AlgorithmRunner / run_batch() reuse one instance across runs."""
    n = 300
    prices = _diffusion_with_jumps(n, 1e-5, {150: 15.0}, seed=5)

    inst = get_algorithm(ALGO_NAME)
    out1 = pd.DataFrame([inst.step({"raw_midprice": float(p)}) for p in prices])

    inst.reset()
    out2 = pd.DataFrame([inst.step({"raw_midprice": float(p)}) for p in prices])

    for col in EXPECTED_KEYS:
        v1, v2 = out1[col].values, out2[col].values
        assert np.array_equal(np.isnan(v1), np.isnan(v2)), (
            f"reset() changed the NaN pattern for {col}"
        )
        mask = np.isfinite(v1) & np.isfinite(v2)
        np.testing.assert_allclose(
            v1[mask], v2[mask], rtol=1e-9, atol=1e-12,
            err_msg=f"reset() did not restore identical output for {col}",
        )


# ══════════════════════════════════════════════════════════════════════════
#  3. run_batch() / step() parity
# ══════════════════════════════════════════════════════════════════════════

def test_run_batch_step_parity():
    """Plants: the vectorized run_batch() path must agree with the
    tick-by-tick step() loop (nan-aware, rtol=1e-9) on a 5,000-tick series
    with planted jumps, past the warmup window — mirrors
    TestStepBatchConsistency in test_algorithms.py but at tighter tolerance
    since this is a deterministic synthetic series, not real data."""
    n = 5000
    sigma = 1e-5
    jump_indices = [500, 1500, 2500, 3500, 4500]
    prices = _diffusion_with_jumps(n, sigma, {idx: 15.0 for idx in jump_indices}, seed=3)
    df = pd.DataFrame({"raw_midprice": prices})

    inst_batch = get_algorithm(ALGO_NAME)
    batch_result = inst_batch.run_batch(df)

    inst_step = get_algorithm(ALGO_NAME)
    step_result = pd.DataFrame(
        [inst_step.step({"raw_midprice": float(p)}) for p in prices]
    )

    warmup = inst_batch.warmup
    assert warmup > 0
    for col in EXPECTED_KEYS:
        b = batch_result[col].values[warmup:]
        s = step_result[col].values[warmup:]
        assert np.array_equal(np.isnan(b), np.isnan(s)), (
            f"batch vs step NaN pattern mismatch in {col}"
        )
        mask = np.isfinite(b) & np.isfinite(s)
        assert mask.sum() > 0, f"no overlapping finite values for {col}"
        np.testing.assert_allclose(
            b[mask], s[mask], rtol=1e-9, atol=1e-12,
            err_msg=f"batch vs step mismatch in {col}",
        )


def test_warmup_blanking():
    """Plants: the first `warmup` rows of run_batch() output are entirely
    NaN (base.py's run_batch() NaN-blanks `result_df.iloc[:warmup]`)."""
    inst = get_algorithm(ALGO_NAME, window=100)
    warmup = inst.warmup
    assert warmup > 0, "expected a nonzero declared warmup (>= window)"

    n = max(warmup * 3, 500)
    prices = _diffusion_with_jumps(n, 1e-5, {}, seed=9)
    df = pd.DataFrame({"raw_midprice": prices})
    result = inst.run_batch(df)

    assert result.iloc[:warmup].isna().all().all(), (
        "warmup rows are not fully NaN-blanked"
    )


# ══════════════════════════════════════════════════════════════════════════
#  4/5. Planted EVT detection + fixed-mode superset sanity
# ══════════════════════════════════════════════════════════════════════════

_EVT_N = 5000
_EVT_SIGMA = 1e-5
_EVT_JUMP_INDICES = [500, 1500, 2500, 3500, 4500]  # well separated, > window apart
_EVT_JUMP_MULT = 15.0  # 15*sigma, far above any sane threshold (~6-9 for EVT)
_EVT_ALPHA = 0.01
_EVT_SEED = 21


def _build_evt_series() -> np.ndarray:
    return _diffusion_with_jumps(
        _EVT_N, _EVT_SIGMA,
        {idx: _EVT_JUMP_MULT for idx in _EVT_JUMP_INDICES},
        seed=_EVT_SEED,
    )


def test_planted_evt_detection():
    """Plants: with threshold_mode='evt' and evt_block == series length,
    EVERY planted 15-sigma jump must be flagged, and false positives on the
    pure-diffusion remainder must stay within the EVT-implied false-alarm
    budget: max(2, ceil(3 * alpha * n_clean)). Also asserts evt_threshold()
    is materially stricter than the legacy fixed c=3.0 (Lee-Mykland 2008
    Gumbel asymptotics give ~6-9 at these block sizes, not 3)."""
    prices = _build_evt_series()
    df = pd.DataFrame({"raw_midprice": prices})

    inst = get_algorithm(
        ALGO_NAME, window=100, threshold_mode="evt",
        evt_alpha=_EVT_ALPHA, evt_block=_EVT_N,
    )
    assert inst.evt_threshold() > 3.0, (
        "EVT threshold should be materially stricter than the legacy fixed c=3.0"
    )

    result = inst.run_batch(df)
    detected = result["alg_jd2_detected"].values
    warmup = inst.warmup

    for idx in _EVT_JUMP_INDICES:
        assert detected[idx] == 1.0, (
            f"planted 15-sigma jump at {idx} not detected "
            f"(L={result['alg_jd2_statistic'].iloc[idx]})"
        )

    is_jump = np.zeros(_EVT_N, dtype=bool)
    is_jump[_EVT_JUMP_INDICES] = True
    valid = np.isfinite(detected)
    valid[:warmup] = False
    clean_mask = valid & ~is_jump

    n_clean = int(clean_mask.sum())
    false_positives = int(((detected == 1.0) & clean_mask).sum())
    fp_budget = max(2, int(np.ceil(3 * _EVT_ALPHA * n_clean)))

    assert false_positives <= fp_budget, (
        f"{false_positives} false positives exceed EVT budget {fp_budget} "
        f"(n_clean={n_clean}, alpha={_EVT_ALPHA})"
    )


def test_fixed_mode_is_superset_of_evt_detections():
    """Plants: threshold_mode='fixed' with significance=3.0 is a LOOSER cut
    than the EVT threshold (~6-9 at this block size), so every EVT
    detection must also fire under fixed mode (superset), on the identical
    series."""
    prices = _build_evt_series()
    df = pd.DataFrame({"raw_midprice": prices})

    evt_inst = get_algorithm(
        ALGO_NAME, window=100, threshold_mode="evt",
        evt_alpha=_EVT_ALPHA, evt_block=_EVT_N,
    )
    fixed_inst = get_algorithm(
        ALGO_NAME, window=100, threshold_mode="fixed", significance=3.0,
    )

    evt_detected = evt_inst.run_batch(df)["alg_jd2_detected"].values
    fixed_detected = fixed_inst.run_batch(df)["alg_jd2_detected"].values

    evt_idx = set(np.where(evt_detected == 1.0)[0].tolist())
    fixed_idx = set(np.where(fixed_detected == 1.0)[0].tolist())

    assert evt_idx.issubset(fixed_idx), (
        f"fixed(sig=3.0) should be a superset of evt detections; "
        f"evt-only indices: {sorted(evt_idx - fixed_idx)}"
    )
    assert len(fixed_idx) >= len(evt_idx)


# ══════════════════════════════════════════════════════════════════════════
#  6. Staggered (skip-one) bipower variation robustness
# ══════════════════════════════════════════════════════════════════════════

def test_staggered_bv_inflates_second_of_two_consecutive_jumps():
    """Plants: two consecutive same-sign 15-sigma jumps at ticks j, j+1.
    Under the naive adjacent-pair bipower estimator (staggered_bv=False),
    the bv estimate feeding L(j+1) is computed from the return buffer that
    still includes r_j (the first jump) as its most recent adjacent-product
    term, inflating bv and suppressing L(j+1). Staggered_bv=True skips that
    immediate-neighbor pairing, so L(j+1) must come out LARGER — i.e. the
    second jump is easier to detect immediately after a first one."""
    n = 2000
    sigma = 1e-5
    j = 1000
    prices = _diffusion_with_jumps(n, sigma, {j: 15.0, j + 1: 15.0}, seed=7)
    df = pd.DataFrame({"raw_midprice": prices})

    inst_stag = get_algorithm(
        ALGO_NAME, window=100, staggered_bv=True,
        threshold_mode="fixed", significance=3.0,
    )
    inst_plain = get_algorithm(
        ALGO_NAME, window=100, staggered_bv=False,
        threshold_mode="fixed", significance=3.0,
    )

    L_stag = inst_stag.run_batch(df)["alg_jd2_statistic"].iloc[j + 1]
    L_plain = inst_plain.run_batch(df)["alg_jd2_statistic"].iloc[j + 1]

    assert np.isfinite(L_stag) and np.isfinite(L_plain)
    assert L_stag > L_plain, (
        f"staggered_bv=True should raise L at the second of two consecutive "
        f"jumps (adjacent-product inflation skipped); got L_stag={L_stag} "
        f"<= L_plain={L_plain}"
    )


# ══════════════════════════════════════════════════════════════════════════
#  7. Directional asymmetry + reversion sign convention
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("direction", ["up", "down"])
def test_directional_reversion_routing(direction: str):
    """Plants: alg_jd2_rev_up/alg_jd2_rev_down must route the single
    reversion signal by the sign of the triggering jump — rev_up nonzero
    and rev_down == 0 after an UP jump, and mirrored after a DOWN jump.
    Also re-asserts v1's sign convention (REV > 0 while genuinely
    reverting) carries over unchanged to alg_jd2_reversion.

    Series construction mirrors test_winning_algos.py::
    test_reversion_sign_convention exactly (flat baseline + isolated jump +
    monotone partial decay), so the sign convention is provably identical
    to the frozen v1 baseline.
    """
    n = 400
    jump_idx = 200
    decay_ticks = 30

    if direction == "up":
        prices = _decaying_jump_series(n, jump_idx, jump_price=101.0,
                                        decay_ticks=decay_ticks, decay_step=0.02)
    else:
        prices = _decaying_jump_series(n, jump_idx, jump_price=99.0,
                                        decay_ticks=decay_ticks, decay_step=0.02)

    df = pd.DataFrame({"raw_midprice": prices})
    inst = get_algorithm(
        ALGO_NAME, window=100, threshold_mode="fixed", significance=3.0,
        adaptive_horizon=False, reversion_horizon=50,
    )
    result = inst.run_batch(df)

    assert result["alg_jd2_detected"].iloc[jump_idx] == 1.0, "planted jump not detected"
    magnitude = result["alg_jd2_magnitude"].iloc[jump_idx]
    if direction == "up":
        assert magnitude > 0
    else:
        assert magnitude < 0

    window = result.iloc[jump_idx + 1: jump_idx + decay_ticks]
    reversion = window["alg_jd2_reversion"]
    rev_up = window["alg_jd2_rev_up"]
    rev_down = window["alg_jd2_rev_down"]

    active = reversion != 0.0
    assert active.sum() > 0, "expected a nonzero reversion signal during tracking"

    # Sign convention (same as v1): price genuinely reverting -> REV > 0.
    assert (reversion[active] > 0).all(), (
        f"reversion should be positive while reverting: {reversion[active].values}"
    )

    if direction == "up":
        assert (rev_up[active] == reversion[active]).all()
        assert (rev_down[active] == 0.0).all()
    else:
        assert (rev_down[active] == reversion[active]).all()
        assert (rev_up[active] == 0.0).all()


# ══════════════════════════════════════════════════════════════════════════
#  8. Magnitude-adaptive reversion horizon
# ══════════════════════════════════════════════════════════════════════════

def _adaptive_horizon_series():
    """Two isolated planted jumps (8-sigma, 30-sigma), far apart, each
    followed by: one tiny deterministic counter-step, then an EXACT price
    freeze (zero returns). This pins the reversion value at a nonzero
    CONSTANT for every tick strictly inside the tracking horizon and at
    exactly 0.0 the instant the horizon expires (per contract: 'else 0.0'),
    which makes "count of nonzero-reversion ticks" an exact, deterministic
    measurement of the horizon length actually used by the implementation
    — with zero risk of a spurious extra detection contaminating the count
    (freezing forces all subsequent returns to be exactly 0).

    A `window`-sized stretch of ordinary diffusion is re-inserted
    immediately before the second (large) jump so its bipower-variation
    estimate is calibrated the same way as the first jump's, rather than
    against an artificially frozen (near-zero) local volatility.
    """
    sigma = 1e-5
    window = 100
    n = 3000
    idx_small, mult_small = 300, 8.0
    idx_large, mult_large = 1500, 30.0
    counter_frac = 0.02  # small deterministic counter-step, well below any threshold

    rng = np.random.default_rng(11)
    rets = rng.normal(0.0, sigma, n)
    rets[0] = 0.0

    rets[idx_small] = mult_small * sigma
    rets[idx_small + 1] = -counter_frac * mult_small * sigma
    freeze_start = idx_small + 2
    freeze_end = idx_large - window - 10  # re-open `window`+10 ticks of diffusion
    assert freeze_end > freeze_start, "series too short for the planted layout"
    rets[freeze_start:freeze_end] = 0.0

    rets[idx_large] = mult_large * sigma
    rets[idx_large + 1] = -counter_frac * mult_large * sigma
    rets[idx_large + 2:] = 0.0  # frozen for the remainder of the series

    prices = 100.0 * np.exp(np.cumsum(rets))
    return prices, idx_small, idx_large, window


def test_adaptive_horizon_scales_with_jump_magnitude():
    """Plants: with adaptive_horizon=True, a 30-sigma jump must produce a
    STRICTLY LONGER nonzero-reversion tracking window than an 8-sigma jump
    (bigger L(t) at detection -> longer horizon), both capped at
    horizon_max; with adaptive_horizon=False both must equal
    reversion_horizon EXACTLY regardless of jump size."""
    prices, idx_small, idx_large, window = _adaptive_horizon_series()
    df = pd.DataFrame({"raw_midprice": prices})
    horizon_max = 500
    reversion_horizon = 50

    def _count_nonzero_after(result: pd.DataFrame, idx: int, end: int) -> int:
        rev = result["alg_jd2_reversion"].iloc[idx + 1: end].values
        return int(np.sum(rev != 0.0))

    # --- adaptive_horizon=True ---
    inst_adaptive = get_algorithm(
        ALGO_NAME, window=window, threshold_mode="fixed", significance=3.0,
        adaptive_horizon=True, reversion_horizon=reversion_horizon,
        horizon_per_l=10.0, horizon_max=horizon_max,
    )
    result_adaptive = inst_adaptive.run_batch(df)

    n = len(prices)
    small_end = idx_large - window - 10  # boundary of the small jump's isolation window
    count_small_adaptive = _count_nonzero_after(result_adaptive, idx_small, small_end)
    count_large_adaptive = _count_nonzero_after(result_adaptive, idx_large, n)

    assert count_large_adaptive > count_small_adaptive, (
        f"adaptive horizon should grow with jump magnitude: "
        f"large={count_large_adaptive} <= small={count_small_adaptive}"
    )
    assert count_small_adaptive <= horizon_max
    assert count_large_adaptive <= horizon_max

    # --- adaptive_horizon=False ---
    inst_fixed = get_algorithm(
        ALGO_NAME, window=window, threshold_mode="fixed", significance=3.0,
        adaptive_horizon=False, reversion_horizon=reversion_horizon,
        horizon_max=horizon_max,
    )
    result_fixed = inst_fixed.run_batch(df)

    count_small_fixed = _count_nonzero_after(result_fixed, idx_small, small_end)
    count_large_fixed = _count_nonzero_after(result_fixed, idx_large, n)

    assert count_small_fixed == reversion_horizon, (
        f"non-adaptive horizon must equal reversion_horizon exactly, "
        f"got {count_small_fixed} for the small jump"
    )
    assert count_large_fixed == reversion_horizon, (
        f"non-adaptive horizon must equal reversion_horizon exactly, "
        f"got {count_large_fixed} for the large jump"
    )


# ══════════════════════════════════════════════════════════════════════════
#  10. EVT threshold closed-form + monotonicity
# ══════════════════════════════════════════════════════════════════════════

def _evt_threshold_reference(n: float, alpha: float) -> float:
    """Independent reference implementation of the Lee-Mykland (2008)
    Gumbel-asymptotic critical value, transcribed directly from the spec —
    used to pin down evt_threshold() against the exact closed form rather
    than just checking loose bounds."""
    c = np.sqrt(2.0 / np.pi)
    ln_n = np.log(n)
    C_n = (np.sqrt(2.0 * ln_n) / c
           - (np.log(np.pi) + np.log(ln_n)) / (2.0 * c * np.sqrt(2.0 * ln_n)))
    S_n = 1.0 / (c * np.sqrt(2.0 * ln_n))
    beta_star = -np.log(-np.log(1.0 - alpha))
    return C_n + S_n * beta_star


def test_evt_threshold_matches_closed_form():
    """Plants: evt_threshold() must equal the exact Gumbel closed form for
    evt_block=864000 (1 day @ 100ms), evt_alpha=0.01, and land in (6.0, 9.0)
    as stated in the spec (this is materially stricter than the legacy
    fixed c=3.0)."""
    inst = get_algorithm(ALGO_NAME, evt_block=864000, evt_alpha=0.01)
    expected = _evt_threshold_reference(864000, 0.01)
    assert abs(inst.evt_threshold() - expected) < 1e-9
    assert 6.0 < inst.evt_threshold() < 9.0


def test_evt_threshold_monotone_in_block_size():
    """Plants: threshold(n) is increasing in n at fixed alpha (bigger
    families of draws need a more extreme critical value to hold the same
    family-wise false-alarm rate)."""
    small = get_algorithm(ALGO_NAME, evt_block=10_000, evt_alpha=0.01)
    large = get_algorithm(ALGO_NAME, evt_block=1_000_000, evt_alpha=0.01)
    assert large.evt_threshold() > small.evt_threshold()


def test_evt_threshold_monotone_in_alpha():
    """Plants: threshold(alpha) is decreasing in alpha at fixed n (a
    smaller tail probability demands a stricter/higher cutoff)."""
    loose = get_algorithm(ALGO_NAME, evt_block=864000, evt_alpha=0.05)
    strict = get_algorithm(ALGO_NAME, evt_block=864000, evt_alpha=0.001)
    assert strict.evt_threshold() > loose.evt_threshold()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
