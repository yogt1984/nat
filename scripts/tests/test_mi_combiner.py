"""PROC-3 — synergy-aware MI combiner. Two things must be true, and one must not.

MUST: it finds information that greedy selection structurally cannot. `greedy_select`
starts at `argmax_f I(f;y)` and extends conditionally, so a *synergistic* pair — two
features each carrying ~0 bits alone but the label jointly (XOR) — is invisible to it:
neither feature can ever be step 1. The spec's acceptance test is exactly this, and the
greedy baseline is asserted to fail alongside, so the improvement is documented rather
than claimed.

MUST NOT: manufacture bits. A GBDT handed 3 features and a shuffled label will fit it
perfectly in-sample, so an in-fold `combo` column would carry enormous MI about pure
noise. The combiner is therefore cross-fit with purged folds, and the decisive test here
feeds it a shuffled label and asserts the emitted combo does NOT clear the PROC-12 null.
If that test ever goes green-by-accident, every downstream finding built on a combo is
worthless — it is the single most important assertion in this file.

Also covered: redundancy (a duplicated feature must not be selected twice), the
TransformProcess contract (index, no mutation, chainability), determinism under a seed,
and imported gates (null z from it_engine.toml, never a literal here).
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
              horizons={"h": 1}, costs={})
    kw.update(over)
    return ProcessContext(**kw)


def _synergy_frame(n=3000, seed=11, noise=0.25) -> pd.DataFrame:
    """XOR plant: y depends on sign(f1)·sign(f2); neither f1 nor f2 alone informs it.

    f3/f4 are pure noise. The label is exposed as a column (`tb_label`) so the process
    runs in PROC-17 label mode — no price series is involved and the plant is exact.
    """
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = np.sign(f1) * np.sign(f2) + rng.normal(scale=noise, size=n)
    return pd.DataFrame({
        "raw_midprice": 100 + np.cumsum(rng.normal(scale=0.01, size=n)),
        "feat_a": f1, "feat_b": f2,
        "feat_noise1": rng.normal(size=n), "feat_noise2": rng.normal(size=n),
        "tb_label": y,
    })


def _run(df, ctx=None, **params):
    from processes.mi_combiner import MICombinerProcess
    ctx = ctx or _ctx(target_col="tb_label")
    return MICombinerProcess(**params).transform(df, ctx)


# ── the plant is what we think it is ─────────────────────────────────────────────
class TestThePlantIsSynergistic:
    def test_each_feature_alone_carries_almost_nothing(self):
        from it_engine.estimators import ksg_mi
        df = _synergy_frame()
        y = df["tb_label"].to_numpy()
        for col in ("feat_a", "feat_b"):
            assert ksg_mi(df[col].to_numpy(), y, k=5) < 0.05, f"{col} leaks marginally"

    def test_the_pair_jointly_carries_a_lot(self):
        """Chain rule: I((a,b);y) = I(a;y) + I(b;y|a). The second term is the synergy."""
        from it_engine.estimators import cmi, ksg_mi
        df = _synergy_frame()
        y = df["tb_label"].to_numpy()
        a, b = df["feat_a"].to_numpy(), df["feat_b"].to_numpy()
        joint = ksg_mi(a, y, k=5) + cmi(b, y, a, k=5)
        assert joint > 0.15, f"planted synergy too weak to test against ({joint:.3f})"


# ── MUST: beat the myopic baseline ───────────────────────────────────────────────
class TestSynergyRecovery:
    def test_greedy_baseline_misses_the_pair(self):
        """Documents the gap PROC-3 exists to close — not a criticism of greedy_select."""
        from it_engine.feature_selector import greedy_select
        df = _synergy_frame()
        feats = {c: df[c].to_numpy() for c in
                 ("feat_a", "feat_b", "feat_noise1", "feat_noise2")}
        picked = greedy_select(feats, df["tb_label"].to_numpy(), fee_rt_bps=0.0,
                               sigma_r_bps=100.0, max_features=2, k=5)
        names = [p["name"] for p in picked]
        assert set(names) != {"feat_a", "feat_b"} or all(
            p["mi"] < 0.05 for p in picked if p["name"] in ("feat_a", "feat_b")), (
            f"greedy unexpectedly recovered the synergistic pair first: {names}")

    def test_combiner_selects_the_synergistic_pair(self):
        derived, result = _run(_synergy_frame())
        selected = set(result.summary["selected"])
        assert {"feat_a", "feat_b"} <= selected, f"selected {selected}"

    def test_combo_carries_information_above_null(self):
        from it_engine.null_calibration import load_null_config
        derived, result = _run(_synergy_frame())
        f = [x for x in result.findings if x.feature == result.summary["combo_column"]][0]
        assert f.extras["z"] >= load_null_config()["null_z_threshold"]
        assert f.extras["bits_above_null"] > 0.05, f.extras

    def test_combo_beats_the_best_single_feature(self):
        """Spec acceptance: the combo's null-calibrated bits exceed the best single's."""
        derived, result = _run(_synergy_frame())
        s = result.summary
        assert s["combo_bits_above_null"] > s["best_single_bits_above_null"], s


# ── MUST NOT: manufacture bits ───────────────────────────────────────────────────
class TestNoLeakage:
    def test_shuffled_label_yields_a_combo_at_the_null(self):
        """The decisive test. In-fold predictions would print huge MI on pure noise."""
        from it_engine.null_calibration import load_null_config
        df = _synergy_frame()
        rng = np.random.default_rng(4)
        df["tb_label"] = rng.permutation(df["tb_label"].to_numpy())
        derived, result = _run(df)
        combo_findings = [x for x in result.findings
                          if x.feature == result.summary["combo_column"]]
        if not combo_findings:                      # refusing to emit is also a pass
            return
        f = combo_findings[0]
        z = f.extras["z"]
        assert z < load_null_config()["null_z_threshold"], (
            f"combo on a SHUFFLED label cleared the null (z={z:.2f}) — the cross-fit "
            "is leaking and every combo finding downstream is worthless")
        assert not f.informative

    def test_pure_noise_features_yield_no_combo_edge(self):
        from it_engine.null_calibration import load_null_config
        rng = np.random.default_rng(9)
        n = 3000
        df = pd.DataFrame({"raw_midprice": 100 + np.cumsum(rng.normal(scale=0.01, size=n)),
                           "feat_a": rng.normal(size=n), "feat_b": rng.normal(size=n),
                           "feat_noise1": rng.normal(size=n),
                           "tb_label": rng.normal(size=n)})
        derived, result = _run(df)
        fs = [x for x in result.findings if x.feature == result.summary["combo_column"]]
        if fs:
            assert fs[0].extras["z"] < load_null_config()["null_z_threshold"]

    def test_folds_are_purged_and_coverage_is_complete(self):
        """Purging removes rows from each TRAINING set (Lopez de Prado), so every valid
        row still receives an out-of-fold prediction — coverage complete, training
        neighbourhoods excluded."""
        df = _synergy_frame()
        derived, result = _run(df)
        s = result.summary
        assert s["n_folds"] >= 2
        assert s["purge_bars"] > 0, "purged K-fold requires a purge gap"
        # each interior boundary drops ~2*purge training rows; edges drop ~purge
        assert s["purged_train_rows"] >= s["purge_bars"] * s["n_folds"], (
            f"purging removed only {s['purged_train_rows']} training rows — the folds "
            "are touching and adjacent autocorrelated rows leak into training")
        v = derived[s["combo_column"]].to_numpy()
        assert np.isfinite(v).sum() > 0.9 * len(df), (
            "out-of-fold coverage should span the frame; large NaN gaps mean folds "
            "silently failed to fit")

    def test_a_zero_purge_configuration_is_visibly_different(self):
        """Guards the guard: if purge_bars stopped taking effect, this would go quiet."""
        _, wide = _run(_synergy_frame(), purge_bars=200)
        _, thin = _run(_synergy_frame(), purge_bars=10)
        assert wide.summary["purged_train_rows"] > thin.summary["purged_train_rows"]


# ── redundancy ───────────────────────────────────────────────────────────────────
class TestRedundancy:
    def test_a_duplicated_feature_is_not_selected_twice(self):
        df = _synergy_frame()
        df["feat_a_copy"] = df["feat_a"] + 1e-9 * np.random.default_rng(2).normal(size=len(df))
        derived, result = _run(df)
        sel = result.summary["selected"]
        assert not ({"feat_a", "feat_a_copy"} <= set(sel)), (
            f"selected a feature and its duplicate: {sel}")


# ── contract ─────────────────────────────────────────────────────────────────────
class TestTransformContract:
    def test_registered(self):
        from processes.registry import list_processes
        assert "mi_combiner" in list_processes()

    def test_derived_shares_the_index_and_does_not_mutate_input(self):
        df = _synergy_frame()
        before = df.copy()
        derived, result = _run(df)
        pd.testing.assert_index_equal(derived.index, df.index)
        pd.testing.assert_frame_equal(df, before)

    def test_derived_column_is_chainable_and_named(self):
        derived, result = _run(_synergy_frame())
        col = result.summary["combo_column"]
        assert col in derived.columns and col.startswith("combo_")
        assert np.isfinite(derived[col].to_numpy()).sum() > 0

    def test_missing_target_is_reported_not_crashed(self):
        derived, result = _run(_synergy_frame(), ctx=_ctx(target_col="nope"))
        assert result.summary.get("error") and "nope" in result.summary["error"]
        assert derived.empty or result.summary.get("combo_column") is None

    def test_all_nan_feature_is_skipped_with_a_reason(self):
        df = _synergy_frame()
        df["feat_dead"] = np.nan
        derived, result = _run(df)
        assert any(s["feature"] == "feat_dead" for s in result.features_skipped)


class TestDeterminism:
    def test_same_seed_same_combo(self):
        df = _synergy_frame()
        a, _ = _run(df, seed=7)
        b, _ = _run(df, seed=7)
        col = [c for c in a.columns if c.startswith("combo_")][0]
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(),
                                   equal_nan=True, rtol=0, atol=0)


class TestImportedGates:
    def test_null_threshold_is_not_a_literal_in_the_module(self):
        from processes import mi_combiner
        src = Path(mi_combiner.__file__).read_text()
        assert "= 3.0" not in src, "null z threshold hardcoded — import it from config"

    def test_uses_the_proc17_target_node(self):
        from processes import mi_combiner
        src = Path(mi_combiner.__file__).read_text()
        assert "resolve_targets" in src, "target resolution must go through PROC-17"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
