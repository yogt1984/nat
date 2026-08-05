"""PROC-3: synergy-aware MI combiner — the nonlinear composition greedy selection cannot find.

`it_engine.feature_selector.greedy_select` maximises *marginal* CMI gain. Step 1 is
`argmax_f I(f;y)`, so a **synergistic** pair — two features each carrying ~0 bits alone but
the label jointly (the XOR case) — is structurally invisible to it: neither feature can ever
be chosen first, and the conditional step never gets the chance. `interaction_info` has been
*computed* in `mi_ksg` since the beginning but never *selected on*, and `pca_combo` is only
the linear composition.

This process closes that gap in three parts.

**Selection is pair-aware.** By the chain rule `I((a,b);y) = I(a;y) + I(b;y|a)`, so the
joint information of a pair is computable with the estimators already in the tree. The seed
of the selected set is the best *pair* whenever it beats the best single — which is exactly
the XOR case — and the set is then extended greedily, conditioning on the whole selected set
(`cmi` accepts multivariate `z`, so this is the true conditional gain, not a proxy).

**Redundancy is penalised, not just ignored.** A candidate's gain is discounted by
`lambda · max(0, -II)` where `II = I(f;y|S) - I(f;y)`: negative interaction information means
the candidate re-tells what the set already knows. A duplicated column therefore fails to
earn its place instead of quietly consuming a slot.

**The fit is cross-fit with purged folds, and this is not optional.** A GBDT handed a
handful of features and a shuffled label fits it perfectly in-sample, so an in-fold `combo`
column would carry enormous mutual information about pure noise — the most efficient false
discovery generator this codebase could contain. Folds are contiguous (the data is a time
series, so random K-fold would leak across autocorrelation), each training set is purged by
`purge_bars` on both sides of its held-out block, and purged rows are emitted as NaN rather
than backfilled with an in-fold value. `tests/test_mi_combiner.py` asserts that a shuffled
label produces a combo that does *not* clear the PROC-12 null; if that assertion ever fails,
every downstream finding built on a combo is worthless.

Scoring is null-calibrated (PROC-12) and reported against the best single feature's
null-calibrated bits — the spec's acceptance criterion. The target comes from the PROC-17
node, so label mode brings its own leakage set (a barrier label's siblings are never
candidates) and its own gate.

Emits `combo_mi` as a first-class derived column, chainable into any evaluation process
(`--score-with mi_ksg` / `ic_horizon`) or compilable by PROC-1 once it earns a polarity.

Spec: `docs/specs/process_layer.md` §3.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from it_engine.estimators import cmi, ksg_mi
from it_engine.null_calibration import load_null_config, null_calibrate

from .base import (
    Finding, ProcessContext, ProcessResult, TransformProcess, make_run_id,
    partition_usable_columns,
)
from .registry import register
from .targets import TargetNotFound, feature_columns, resolve_targets

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")

#: Name of the emitted composite column.
COMBO_COLUMN = "combo_mi"


@register
class MICombinerProcess(TransformProcess):
    """Synergy-aware feature selection + purged cross-fit GBDT → one composite column."""

    PARAMS = {
        "max_features": (4, "maximum features in the selected set"),
        "max_candidates": (12, "candidates entering the O(n^2) pair search"),
        "k": (5, "KSG nearest neighbours"),
        "n_folds": (4, "contiguous cross-fit folds"),
        "purge_bars": (100, "rows purged either side of each held-out block"),
        "redundancy_lambda": (0.5, "penalty weight on negative interaction info"),
        "max_samples": (4000, "subsample cap for the estimators"),
        "min_obs": (200, "minimum jointly-valid observations"),
        "null_shuffles": (None, "permutation draws (default: it_engine.toml)"),
        "seed": (42, "seed for subsampling, folds and the null"),
        "target_col": (None, "label column replacing forward returns (PROC-17)"),
    }

    def name(self) -> str:
        return "mi_combiner"

    # ── the process ──────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame, ctx: ProcessContext
                  ) -> tuple[pd.DataFrame, ProcessResult]:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        empty = pd.DataFrame(index=df.index)

        # 1. target (PROC-17 owns resolution, usability and the leakage set)
        try:
            target = resolve_targets(df, ctx, self.params)[0]
        except TargetNotFound as exc:
            return empty, result.finalize(time.time() - t0, error=str(exc))
        y_full = target.values(df)

        # 2. candidates
        cand = feature_columns(df, [
            c for c in df.columns
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
            and pd.api.types.is_numeric_dtype(df[c])
        ], target)
        min_obs = int(self.params["min_obs"])
        usable, skipped = partition_usable_columns(df, cand, min_obs=min_obs)
        result.features_tested = usable
        result.features_skipped = skipped
        if len(usable) < 2:
            return empty, result.finalize(time.time() - t0,
                                          error="need >= 2 usable features to combine")

        rng = np.random.default_rng(int(self.params["seed"]))
        k = int(self.params["k"])
        X_full = df[usable].to_numpy(dtype=np.float64, na_value=np.nan)
        valid = np.isfinite(X_full).all(axis=1) & np.isfinite(y_full)
        if int(valid.sum()) < min_obs:
            return empty, result.finalize(
                time.time() - t0, error=f"jointly valid rows {int(valid.sum())} < {min_obs}")

        idx = np.flatnonzero(valid)
        sub = self._subsample(idx, int(self.params["max_samples"]), rng)
        Xs, ys = X_full[sub], y_full[sub]

        # 3. selection
        selected, sel_trace, best_single = self._select(Xs, ys, usable, k)
        if not selected:
            return empty, result.finalize(time.time() - t0, error="no informative candidates")

        # 4. purged cross-fit (never in-fold)
        combo_full, n_folds, purge, n_purged = self._cross_fit(
            X_full[:, [usable.index(c) for c in selected]], y_full, valid, rng)

        # 5. null-calibrated scoring of the composite
        cfg = load_null_config()
        n_shuffles = int(self.params["null_shuffles"] or cfg["n_shuffles"])
        z_thr, i_min = float(cfg["null_z_threshold"]), float(cfg["i_min"])
        m = np.isfinite(combo_full) & np.isfinite(y_full)
        combo_nr = None
        if int(m.sum()) >= min_obs:
            cm = self._subsample(np.flatnonzero(m), int(self.params["max_samples"]), rng)
            combo_nr = null_calibrate(lambda a, b: ksg_mi(a, b, k=k),
                                      _rank01(combo_full[cm]), _rank01(y_full[cm]),
                                      n_shuffles=n_shuffles, rng=rng)

        best_nr = None
        if best_single is not None:
            col = X_full[:, usable.index(best_single["feature"])]
            mm = np.isfinite(col) & np.isfinite(y_full)
            bs = self._subsample(np.flatnonzero(mm), int(self.params["max_samples"]), rng)
            best_nr = null_calibrate(lambda a, b: ksg_mi(a, b, k=k),
                                     _rank01(col[bs]), _rank01(y_full[bs]),
                                     n_shuffles=n_shuffles, rng=rng)

        # 6. findings + derived column
        if combo_nr is not None:
            result.findings.append(Finding(
                feature=COMBO_COLUMN, horizon=target.horizon_name, metric="mi_bits",
                value=round(combo_nr.raw_bits, 6), p_value=round(combo_nr.p, 6),
                threshold=z_thr,
                informative=bool(combo_nr.informative(i_min=i_min, z_threshold=z_thr)),
                extras={"gate": target.gate, "target": target.label_def,
                        "bits_above_null": round(combo_nr.bits_above_null, 6),
                        "z": round(combo_nr.z, 3), "p": round(combo_nr.p, 6),
                        "null_mean": round(combo_nr.null_mean, 6),
                        "selected": selected, "n_samples": int(m.sum())},
            ))
        for step in sel_trace:
            result.findings.append(Finding(
                feature=step["feature"], horizon=target.horizon_name,
                metric="cmi_gain_bits", value=round(step["gain"], 6),
                informative=False,
                extras={k2: step[k2] for k2 in ("step", "marginal_mi", "raw_gain",
                                                "redundancy_penalty", "reason")},
            ))

        derived = pd.DataFrame({COMBO_COLUMN: combo_full}, index=df.index)
        result.finalize(time.time() - t0)
        result.summary.update({
            "combo_column": COMBO_COLUMN,
            "selected": selected,
            "n_selected": len(selected),
            "seed_kind": sel_trace[0]["reason"] if sel_trace else None,
            "best_single": best_single["feature"] if best_single else None,
            "best_single_bits_above_null": (round(best_nr.bits_above_null, 6)
                                            if best_nr else None),
            "combo_bits_above_null": (round(combo_nr.bits_above_null, 6)
                                      if combo_nr else None),
            "combo_z": round(combo_nr.z, 3) if combo_nr else None,
            "n_folds": n_folds, "purge_bars": purge,
            "purged_train_rows": n_purged,
            "target": target.as_dict(),
            "n_valid": int(valid.sum()),
        })
        return derived, result

    # ── selection ────────────────────────────────────────────────────────────────
    def _select(self, X: np.ndarray, y: np.ndarray, names: list[str], k: int):
        """Pair-aware seed + redundancy-penalised greedy extension.

        Returns (selected names, per-step trace, best single feature record).
        """
        yr = _rank01(y)
        Xr = np.column_stack([_rank01(X[:, j]) for j in range(X.shape[1])])

        marginal = [{"feature": n, "mi": float(ksg_mi(Xr[:, j], yr, k=k))}
                    for j, n in enumerate(names)]
        marginal.sort(key=lambda d: d["mi"], reverse=True)
        best_single = marginal[0] if marginal else None
        if best_single is None:
            return [], [], None

        # candidates for the O(n^2) pair search: the strongest marginals plus, crucially,
        # everything else if the field is small — a synergistic feature ranks LAST here.
        top = [d["feature"] for d in marginal[:int(self.params["max_candidates"])]]
        col = {n: j for j, n in enumerate(names)}

        best_pair, best_joint = None, -np.inf
        for a, b in combinations(top, 2):
            # chain rule: I((a,b);y) = I(a;y) + I(b;y|a)
            joint = (next(d["mi"] for d in marginal if d["feature"] == a)
                     + float(cmi(Xr[:, col[b]], yr, Xr[:, col[a]], k=k)))
            if joint > best_joint:
                best_pair, best_joint = (a, b), joint

        trace: list[dict] = []
        if best_pair is not None and best_joint > best_single["mi"]:
            selected = list(best_pair)
            trace.append({"step": 0, "feature": best_pair[0], "gain": best_joint,
                          "marginal_mi": next(d["mi"] for d in marginal
                                              if d["feature"] == best_pair[0]),
                          "raw_gain": best_joint, "redundancy_penalty": 0.0,
                          "reason": f"pair_seed with {best_pair[1]} "
                                    f"(joint {best_joint:.4f} > best single "
                                    f"{best_single['mi']:.4f})"})
        else:
            selected = [best_single["feature"]]
            trace.append({"step": 0, "feature": best_single["feature"],
                          "gain": best_single["mi"], "marginal_mi": best_single["mi"],
                          "raw_gain": best_single["mi"], "redundancy_penalty": 0.0,
                          "reason": "single_seed"})

        lam = float(self.params["redundancy_lambda"])
        while len(selected) < int(self.params["max_features"]):
            Z = Xr[:, [col[s] for s in selected]]
            best_c, best_score, best_rec = None, 0.0, None
            for n in names:
                if n in selected:
                    continue
                gain = float(cmi(Xr[:, col[n]], yr, Z, k=k))
                mi_n = next(d["mi"] for d in marginal if d["feature"] == n)
                ii = gain - mi_n                     # interaction information vs the set
                penalty = lam * max(0.0, -ii)        # redundancy discounts the gain
                score = gain - penalty
                if score > best_score:
                    best_c, best_score = n, score
                    best_rec = {"step": len(selected), "feature": n, "gain": score,
                                "marginal_mi": mi_n, "raw_gain": gain,
                                "redundancy_penalty": penalty,
                                "reason": "greedy_extension"}
            if best_c is None:
                break
            selected.append(best_c)
            trace.append(best_rec)
        return selected, trace, best_single

    # ── purged cross-fit ─────────────────────────────────────────────────────────
    def _cross_fit(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray,
                   rng: np.random.Generator):
        """Out-of-fold GBDT predictions over contiguous, purged folds.

        Contiguous (not random) because the rows are a time series; purged because a model
        trained on rows adjacent to its held-out block sees the same autocorrelated
        information. Purged rows stay NaN — never an in-fold value.
        """
        import lightgbm as lgb

        n = len(y)
        n_folds = max(2, int(self.params["n_folds"]))
        purge = max(1, int(self.params["purge_bars"]))
        out = np.full(n, np.nan)
        bounds = np.linspace(0, n, n_folds + 1).astype(int)
        n_purged = 0

        for i in range(n_folds):
            lo, hi = bounds[i], bounds[i + 1]
            test = np.zeros(n, dtype=bool)
            test[lo:hi] = True
            train = ~test
            before = int(train.sum())
            train[max(0, lo - purge):min(n, hi + purge)] = False   # purge both sides
            n_purged += before - int(train.sum())
            tr = train & valid
            te = test & valid
            if tr.sum() < 50 or te.sum() == 0:
                continue
            model = lgb.LGBMRegressor(
                n_estimators=120, num_leaves=15, learning_rate=0.08,
                min_child_samples=30, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.9, random_state=int(self.params["seed"]),
                verbose=-1, deterministic=True, force_row_wise=True,
            )
            model.fit(X[tr], y[tr])
            out[te] = model.predict(X[te])
        return out, n_folds, purge, n_purged

    # ── helpers ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _subsample(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
        if len(idx) <= cap:
            return idx
        return np.sort(rng.choice(idx, size=cap, replace=False))


def _rank01(v: np.ndarray) -> np.ndarray:
    """Copula (rank) transform — KSG's documented noise floor is smaller on ranks."""
    from scipy.stats import rankdata
    x = np.asarray(v, dtype=np.float64)
    return rankdata(x) / (len(x) + 1.0)
