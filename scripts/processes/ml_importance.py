"""
ML importance process — gradient-boosted walk-forward feature evaluation.

Trains an LGBMClassifier on the binary direction of the forward return
(up/down) under an expanding walk-forward split (train on [0, t), test on the
next fold — no lookahead), per horizon:

  - per-feature gain importance, averaged across out-of-sample folds and
    normalized per fold (so folds weight equally)
  - out-of-sample directional accuracy
  - confidence-filtered net PnL: trade only when max(p, 1-p) >= threshold,
    direction sign(p - 0.5), minus the round-trip fee per trade

Informative iff the feature ranks in the top_k by mean importance AND the
fold-pooled confidence-filtered strategy clears costs (net PnL > 0) — a
feature the model leans on inside a strategy that loses money is not
information, it is overfit.

With `target_col` set to a label column (e.g. `tb_label` from the
triple_barrier process), nonzero labels replace the return-sign target.

Re-implements the walk-forward loop of `scripts/phase1_signal_test.py`
(polars + print-driven, not importable) on pandas bars. lightgbm is imported
lazily; if unavailable the result carries a structured error instead of
crashing (`summary.error`).
"""

from __future__ import annotations

import time

import numpy as np

from alpha.screener import compute_forward_returns

from .base import EvaluationProcess, Finding, ProcessContext, ProcessResult, make_run_id, partition_usable_columns
from .registry import register

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")

_LGBM_DEFAULTS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_samples": 50,
}


@register
class MLImportanceProcess(EvaluationProcess):
    """LightGBM walk-forward importance + confidence-filtered net PnL."""

    PARAMS = {
        "features": (None, "list of name prefixes to score; None = all non-meta numeric"),
        "n_splits": (5, "walk-forward folds (expanding train, next-fold test)"),
        "top_k": (20, "importance rank cutoff for the informative gate"),
        "confidence_threshold": (0.60, "min class probability to take a trade"),
        "min_obs": (300, "minimum labeled bars"),
        "lgbm": (None, "dict of LGBMClassifier overrides (merged over defaults)"),
        "target_col": (None, "label column replacing return-sign target"),
        "fee_rt_bps": (None, "round-trip fee override; None = ctx.costs hyperliquid"),
        "seed": (7, "LightGBM deterministic seed"),
    }

    def name(self) -> str:
        return "ml_importance"

    def evaluate(self, bars, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params={
                k: v for k, v in self.params.items()
            },
        )
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return result.finalize(time.time() - t0, error="lightgbm not installed")

        min_obs = int(self.params["min_obs"])
        n_splits = int(self.params["n_splits"])
        conf = float(self.params["confidence_threshold"])
        fee_rt = self.params["fee_rt_bps"]
        if fee_rt is None:
            fee_rt = ctx.costs.get("hyperliquid", {}).get("round_trip_taker_bps", 7.0)
        fee_rt = float(fee_rt)
        lgbm_params = {**_LGBM_DEFAULTS, **(self.params["lgbm"] or {}),
                       "random_state": int(self.params["seed"]), "verbose": -1}

        cols = [
            c for c in self.required_columns(list(bars.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
        ]
        target_col = self.params["target_col"] or ctx.target_col
        if target_col:
            cols = [c for c in cols if c != target_col and not c.startswith("tb_")]
        usable, skipped = partition_usable_columns(bars, cols, min_obs=min_obs)
        result.features_tested = usable
        result.features_skipped = skipped
        if not usable:
            return result.finalize(time.time() - t0, error="no usable features")

        X = bars[usable].to_numpy(dtype=np.float64, na_value=np.nan)
        prices = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)

        # Targets: (horizon name, binary label, fwd return in bps for PnL)
        targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if target_col:
            lab = bars[target_col].to_numpy(dtype=np.float64, na_value=np.nan)
            ret_col = "tb_ret" if "tb_ret" in bars.columns else None
            ret = (
                bars[ret_col].to_numpy(dtype=np.float64, na_value=np.nan)
                if ret_col else np.where(np.isfinite(lab), lab, np.nan)
            )
            mask = np.isfinite(lab) & (lab != 0)
            y = np.where(lab > 0, 1.0, 0.0)
            y[~mask] = np.nan
            targets["label"] = (y, ret * 1e4)
        else:
            for h_name, h_bars in ctx.horizons.items():
                fr = compute_forward_returns(prices, h_bars)
                y = np.where(np.isfinite(fr), (fr > 0).astype(np.float64), np.nan)
                targets[h_name] = (y, fr * 1e4)

        for h_name, (y, ret_bps) in targets.items():
            labeled = np.isfinite(y)
            idx = np.flatnonzero(labeled)
            if len(idx) < max(min_obs, (n_splits + 1) * 50):
                continue

            folds = np.array_split(idx, n_splits + 1)
            fold_importance: list[np.ndarray] = []
            fold_acc: list[float] = []
            trade_pnls: list[float] = []
            n_trades = 0

            for i in range(1, n_splits + 1):
                train_idx = np.concatenate(folds[:i])
                test_idx = folds[i]
                if len(train_idx) < 50 or len(test_idx) < 10:
                    continue
                model = LGBMClassifier(importance_type="gain", **lgbm_params)
                model.fit(X[train_idx], y[train_idx])

                imp = np.asarray(model.feature_importances_, dtype=np.float64)
                total = imp.sum()
                fold_importance.append(imp / total if total > 0 else imp)

                proba = model.predict_proba(X[test_idx])[:, 1]
                fold_acc.append(float(np.mean((proba > 0.5) == (y[test_idx] > 0.5))))

                confident = np.maximum(proba, 1.0 - proba) >= conf
                direction = np.sign(proba - 0.5)
                pnl = direction[confident] * ret_bps[test_idx][confident] - fee_rt
                pnl = pnl[np.isfinite(pnl)]
                trade_pnls.extend(pnl.tolist())
                n_trades += len(pnl)

            if not fold_importance:
                continue
            mean_imp = np.mean(np.vstack(fold_importance), axis=0)
            ranks = np.argsort(-mean_imp)
            rank_of = {usable[j]: r + 1 for r, j in enumerate(ranks)}
            wf_accuracy = float(np.mean(fold_acc)) if fold_acc else 0.0
            net_pnl_bps = float(np.mean(trade_pnls)) if trade_pnls else 0.0
            strategy_profitable = bool(trade_pnls) and net_pnl_bps > 0

            for j, feat in enumerate(usable):
                rank = rank_of[feat]
                result.findings.append(Finding(
                    feature=feat, horizon=h_name, metric="wf_importance",
                    value=round(float(mean_imp[j]), 6),
                    informative=bool(
                        rank <= int(self.params["top_k"]) and strategy_profitable
                    ),
                    extras={
                        "rank": rank,
                        "wf_accuracy": round(wf_accuracy, 4),
                        "n_folds": len(fold_importance),
                        "conf_filtered": {
                            "threshold": conf,
                            "n_trades": n_trades,
                            "net_pnl_bps": round(net_pnl_bps, 3),
                        },
                    },
                ))

        return result.finalize(time.time() - t0)
